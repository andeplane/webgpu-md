import { WebGPUContext, storageBufferEntry, uniformBufferEntry, bufferEntry, workgroupCount } from '../compute/WebGPUContext'
import type { SimulationState } from '../core/SimulationState'
import { Integrator } from './Integrator'
import type { IntegratorConfig } from './Integrator'
import velocityVerletShader from '../shaders/velocityVerlet.wgsl?raw'

/**
 * Velocity Verlet integrator
 * 
 * Two-phase integration:
 * 1. Initial: v += 0.5*dt*f/m, x += dt*v, apply PBC
 * 2. (Force calculation)
 * 3. Final: v += 0.5*dt*f/m
 * 
 * Also tracks displacement from last neighbor list rebuild for smart rebuilding.
 */
export class VelocityVerlet extends Integrator {
  readonly name = 'velocity-verlet'

  // GPU buffers
  private paramsBuffer: GPUBuffer
  private lastRebuildPositionsBuffer: GPUBuffer
  private maxDisplacementSqBuffer: GPUBuffer
  private maxDisplacementStagingBuffer: GPUBuffer

  // Pipelines
  private initialPipeline: GPUComputePipeline
  private finalPipeline: GPUComputePipeline
  private bindGroupLayout: GPUBindGroupLayout
  private bindGroup: GPUBindGroup | null = null

  // Cached box parameters
  private boxLx = 0
  private boxLy = 0
  private boxLz = 0
  private originX = 0
  private originY = 0
  private originZ = 0
  private periodic = [true, true, true]

  // Displacement tracking
  private trackDisplacement = true

  constructor(ctx: WebGPUContext, numAtoms: number, config: IntegratorConfig) {
    super(ctx, numAtoms, config)

    // Create parameter buffer (64 bytes, 4 vec4)
    this.paramsBuffer = ctx.createUniformBuffer(64, 'verlet-params')

    // Create displacement tracking buffers
    this.lastRebuildPositionsBuffer = ctx.createStorageBuffer(
      numAtoms * 3 * 4,
      'last-rebuild-positions'
    )

    this.maxDisplacementSqBuffer = ctx.createStorageBuffer(
      4,
      'max-displacement-sq',
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    )

    this.maxDisplacementStagingBuffer = ctx.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'max-displacement-staging',
    })

    // Create bind group layout
    this.bindGroupLayout = ctx.createBindGroupLayout([
      uniformBufferEntry(0),           // params
      storageBufferEntry(1, false),    // positions (read-write)
      storageBufferEntry(2, false),    // velocities (read-write)
      storageBufferEntry(3, true),     // forces (read-only)
      storageBufferEntry(4, true),     // masses (read-only)
      storageBufferEntry(5, true),     // lastRebuildPositions (read-only)
      storageBufferEntry(6, false),    // maxDisplacementSq (read-write, atomic)
    ], 'verlet-layout')

    // Create pipelines
    this.initialPipeline = ctx.createComputePipeline(
      velocityVerletShader,
      'integrateInitial',
      [this.bindGroupLayout],
      'verlet-initial'
    )

    this.finalPipeline = ctx.createComputePipeline(
      velocityVerletShader,
      'integrateFinal',
      [this.bindGroupLayout],
      'verlet-final'
    )
  }

  /**
   * Update for new simulation box
   */
  updateBox(state: SimulationState): void {
    const [lx, ly, lz] = state.box.dimensions
    this.boxLx = lx
    this.boxLy = ly
    this.boxLz = lz
    this.originX = state.box.origin.x
    this.originY = state.box.origin.y
    this.originZ = state.box.origin.z
    this.periodic = [...state.box.periodic]
    this.updateParams()
    this.bindGroup = null  // Force recreation
  }

  /**
   * Enable or disable displacement tracking
   */
  setTrackDisplacement(enabled: boolean): void {
    this.trackDisplacement = enabled
    this.updateParams()
  }

  /**
   * Update parameter buffer
   */
  protected updateParams(): void {
    const buffer = new ArrayBuffer(64)
    const u32View = new Uint32Array(buffer)
    const f32View = new Float32Array(buffer)

    // vec4: numAtoms, halfDt, dt, trackDisplacement
    u32View[0] = this.numAtoms
    f32View[1] = 0.5 * this.dt
    f32View[2] = this.dt
    u32View[3] = this.trackDisplacement ? 1 : 0

    // vec4: originX, originY, originZ, boxLx
    f32View[4] = this.originX
    f32View[5] = this.originY
    f32View[6] = this.originZ
    f32View[7] = this.boxLx

    // vec4: boxLy, boxLz, periodicX, periodicY
    f32View[8] = this.boxLy
    f32View[9] = this.boxLz
    u32View[10] = this.periodic[0] ? 1 : 0
    u32View[11] = this.periodic[1] ? 1 : 0

    // vec4: periodicZ, padding, padding, padding
    u32View[12] = this.periodic[2] ? 1 : 0
    u32View[13] = 0
    u32View[14] = 0
    u32View[15] = 0

    this.ctx.writeBuffer(this.paramsBuffer, new Float32Array(buffer))
  }

  /**
   * Create bind group if needed
   */
  private ensureBindGroup(state: SimulationState): void {
    if (!this.bindGroup) {
      this.bindGroup = this.ctx.createBindGroup(
        this.bindGroupLayout,
        [
          bufferEntry(0, this.paramsBuffer),
          bufferEntry(1, state.positionsBuffer),
          bufferEntry(2, state.velocitiesBuffer),
          bufferEntry(3, state.forcesBuffer),
          bufferEntry(4, state.massesBuffer),
          bufferEntry(5, this.lastRebuildPositionsBuffer),
          bufferEntry(6, this.maxDisplacementSqBuffer),
        ],
        'verlet-bind-group'
      )
    }
  }

  /**
   * Initial integration: v += 0.5*dt*f/m, x += dt*v, PBC
   */
  integrateInitial(state: SimulationState): void {
    this.ensureBindGroup(state)

    this.ctx.runComputePass(
      this.initialPipeline,
      [this.bindGroup!],
      workgroupCount(this.numAtoms, 256),
      1,
      1,
      'integrate-initial'
    )
  }

  /**
   * Final integration: v += 0.5*dt*f/m
   */
  integrateFinal(state: SimulationState): void {
    this.ensureBindGroup(state)

    this.ctx.runComputePass(
      this.finalPipeline,
      [this.bindGroup!],
      workgroupCount(this.numAtoms, 256),
      1,
      1,
      'integrate-final'
    )
  }

  /**
   * Save current positions as the reference for displacement tracking
   * Call this after rebuilding the neighbor list
   */
  savePositionsForRebuild(state: SimulationState): void {
    const commandEncoder = this.ctx.createCommandEncoder('save-positions')
    commandEncoder.copyBufferToBuffer(
      state.positionsBuffer,
      0,
      this.lastRebuildPositionsBuffer,
      0,
      this.numAtoms * 3 * 4
    )
    this.ctx.submit([commandEncoder.finish()])

    // Reset max displacement counter
    this.resetMaxDisplacement()
  }

  /**
   * Reset the max displacement counter to zero
   */
  resetMaxDisplacement(): void {
    const zero = new Uint32Array([0])
    this.ctx.writeBuffer(this.maxDisplacementSqBuffer, zero)
  }

  /**
   * Get the maximum displacement squared since last rebuild
   * Returns the actual squared displacement (unscaled)
   */
  async getMaxDisplacementSq(): Promise<number> {
    // Wait for any pending GPU work to complete before using staging buffer
    // This prevents "buffer used in submit while pending map" errors
    await this.ctx.waitForGPU()

    const commandEncoder = this.ctx.createCommandEncoder()
    commandEncoder.copyBufferToBuffer(
      this.maxDisplacementSqBuffer,
      0,
      this.maxDisplacementStagingBuffer,
      0,
      4
    )
    this.ctx.submit([commandEncoder.finish()])

    await this.maxDisplacementStagingBuffer.mapAsync(GPUMapMode.READ)
    const data = new Uint32Array(this.maxDisplacementStagingBuffer.getMappedRange())
    const scaledValue = data[0]
    this.maxDisplacementStagingBuffer.unmap()

    // Unscale: we multiplied by 1e6 in shader
    return scaledValue / 1000000.0
  }

  /**
   * Check if neighbor list needs to be rebuilt based on displacement
   * Uses the LAMMPS criterion: rebuild when 2 * maxDisplacement > skin
   * (factor of 2 because two atoms could move toward each other)
   * 
   * @param skin - The skin distance used for the neighbor list
   * @returns true if rebuild is needed
   */
  async needsNeighborRebuild(skin: number): Promise<boolean> {
    const maxDispSq = await this.getMaxDisplacementSq()
    const maxDisp = Math.sqrt(maxDispSq)
    
    // LAMMPS criterion: 2 * maxDisplacement > skin
    // This is conservative: two atoms could approach each other
    // LAMMPS criterion: 2 * maxDisplacement > skin
    return 2 * maxDisp > skin
  }

  /**
   * Destroy GPU resources
   */
  destroy(): void {
    this.paramsBuffer.destroy()
    this.lastRebuildPositionsBuffer.destroy()
    this.maxDisplacementSqBuffer.destroy()
    this.maxDisplacementStagingBuffer.destroy()
  }
}
