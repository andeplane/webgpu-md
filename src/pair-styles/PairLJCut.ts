import { WebGPUContext, storageBufferEntry, uniformBufferEntry, bufferEntry, workgroupCount } from '../compute/WebGPUContext'
import type { NeighborList } from '../compute/NeighborList'
import type { SimulationState } from '../core/SimulationState'
import { PairStyle, PairCoeffMatrix } from './PairStyle'
import type { PairStyleConfig, PairCoeff, ForceResult } from './PairStyle'
import pairLJCutShader from '../shaders/pairLJCut.wgsl?raw'

/**
 * LJ coefficients for a type pair
 */
interface LJCoeff extends Record<string, number> {
  epsilon: number
  sigma: number
  cutoff: number  // Per-pair cutoff (optional, defaults to global)
}

/**
 * Lennard-Jones pair style (lj/cut)
 * 
 * Potential: U(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
 * Force: F(r) = 24 * epsilon / r * [2*(sigma/r)^12 - (sigma/r)^6]
 */
export class PairLJCut extends PairStyle {
  readonly name = 'lj/cut'

  // Coefficient matrix
  private coeffMatrix: PairCoeffMatrix<LJCoeff>

  // GPU buffers
  private paramsBuffer: GPUBuffer
  private energyBuffer: GPUBuffer
  private energyStagingBuffer: GPUBuffer

  // Pipelines
  private zeroPipeline: GPUComputePipeline
  private forcePipeline: GPUComputePipeline
  private bindGroupLayout: GPUBindGroupLayout
  private bindGroup: GPUBindGroup | null = null

  // Cached box parameters
  private boxLx = 0
  private boxLy = 0
  private boxLz = 0
  private periodic = [true, true, true]

  constructor(ctx: WebGPUContext, numAtoms: number, config: PairStyleConfig) {
    super(ctx, numAtoms, config)

    this.coeffMatrix = new PairCoeffMatrix<LJCoeff>(config.numTypes)

    // Create parameter buffer (aligned to 16 bytes)
    // Size: 12 u32/f32 values = 48 bytes, padded to 48
    this.paramsBuffer = ctx.createUniformBuffer(48, 'lj-params')

    // Create coefficient buffer
    // Each pair has 8 f32 values (lj1, lj2, lj3, lj4, cutsq, offset, padding x2)
    // Size: numTypes * numTypes * 8 * 4 bytes
    const coeffSize = config.numTypes * config.numTypes * 8 * 4
    this.coeffBuffer = ctx.createStorageBuffer(coeffSize, 'lj-coeffs')

    // Create energy buffer
    this.energyBuffer = ctx.createStorageBuffer(numAtoms * 4, 'lj-energy')
    this.energyStagingBuffer = ctx.device.createBuffer({
      size: numAtoms * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'lj-energy-staging',
    })

    // Create bind group layout
    this.bindGroupLayout = ctx.createBindGroupLayout([
      uniformBufferEntry(0),           // params
      storageBufferEntry(1, true),     // coeffs (read-only)
      storageBufferEntry(2, true),     // positions (read-only)
      storageBufferEntry(3, true),     // types (read-only)
      storageBufferEntry(4, true),     // neighborList (read-only)
      storageBufferEntry(5, true),     // numNeighbors (read-only)
      storageBufferEntry(6, false),    // forces (read-write)
      storageBufferEntry(7, false),    // energy (read-write)
    ], 'lj-layout')

    // Create pipelines
    this.zeroPipeline = ctx.createComputePipeline(
      pairLJCutShader,
      'zeroForces',
      [this.bindGroupLayout],
      'lj-zero'
    )

    this.forcePipeline = ctx.createComputePipeline(
      pairLJCutShader,
      'computeForces',
      [this.bindGroupLayout],
      'lj-force'
    )

    // Initialize coefficients if provided
    if (config.coeffs) {
      this.initCoeffs(config.coeffs)
    }
  }

  /**
   * Set coefficients for type pair (i, j)
   * @param typeI - First atom type (0-indexed)
   * @param typeJ - Second atom type (0-indexed)
   * @param epsilon - LJ well depth
   * @param sigma - LJ size parameter
   * @param cutoff - Optional per-pair cutoff (defaults to global)
   */
  setCoeff(typeI: number, typeJ: number, epsilon: number, sigma: number, cutoff?: number): void {
    this.coeffMatrix.set(typeI, typeJ, {
      epsilon,
      sigma,
      cutoff: cutoff ?? this.cutoff,
    })
    this.uploadCoeffs()
  }

  /**
   * Initialize coefficients from config
   */
  initCoeffs(coeffs: PairCoeff[]): void {
    for (const c of coeffs) {
      this.coeffMatrix.set(c.typeI, c.typeJ, {
        epsilon: c.epsilon ?? 1.0,
        sigma: c.sigma ?? 1.0,
        cutoff: c.cutoff ?? this.cutoff,
      })
    }
    this.uploadCoeffs()
  }

  /**
   * Upload coefficients to GPU
   */
  private uploadCoeffs(): void {
    const data = this.coeffMatrix.packForGPU(8, (coeff) => {
      const sig2 = coeff.sigma * coeff.sigma
      const sig6 = sig2 * sig2 * sig2
      const sig12 = sig6 * sig6

      // LAMMPS-style coefficients
      const lj1 = 48.0 * coeff.epsilon * sig12
      const lj2 = 24.0 * coeff.epsilon * sig6
      const lj3 = 4.0 * coeff.epsilon * sig12
      const lj4 = 4.0 * coeff.epsilon * sig6
      const cutsq = coeff.cutoff * coeff.cutoff

      // Energy offset at cutoff
      const ratio = coeff.sigma / coeff.cutoff
      const ratio6 = Math.pow(ratio, 6)
      const offset = 4.0 * coeff.epsilon * (ratio6 * ratio6 - ratio6)

      return [lj1, lj2, lj3, lj4, cutsq, offset, 0, 0]
    })

    this.ctx.writeBuffer(this.coeffBuffer!, data)
  }

  /**
   * Update box parameters for minimum image
   */
  updateBox(state: SimulationState): void {
    const [lx, ly, lz] = state.box.dimensions
    this.boxLx = lx
    this.boxLy = ly
    this.boxLz = lz
    this.periodic = [...state.box.periodic]
    this.bindGroup = null  // Force recreation
  }

  /**
   * Compute LJ forces
   */
  compute(
    state: SimulationState,
    neighborList: NeighborList,
    computeEnergy = false
  ): ForceResult {
    // Update params
    this.updateParams(neighborList.maxNeighbors, computeEnergy)

    // Create bind group if needed
    if (!this.bindGroup) {
      this.bindGroup = this.ctx.createBindGroup(
        this.bindGroupLayout,
        [
          bufferEntry(0, this.paramsBuffer),
          bufferEntry(1, this.coeffBuffer!),
          bufferEntry(2, state.positionsBuffer),
          bufferEntry(3, state.typesBuffer),
          bufferEntry(4, neighborList.neighborListBuffer),
          bufferEntry(5, neighborList.numNeighborsBuffer),
          bufferEntry(6, state.forcesBuffer),
          bufferEntry(7, this.energyBuffer),
        ],
        'lj-bind-group'
      )
    }

    const commandEncoder = this.ctx.createCommandEncoder('lj-compute')

    // Pass 1: Zero forces
    {
      const pass = commandEncoder.beginComputePass({ label: 'zero-forces' })
      pass.setPipeline(this.zeroPipeline)
      pass.setBindGroup(0, this.bindGroup)
      pass.dispatchWorkgroups(workgroupCount(this.numAtoms, 64))
      pass.end()
    }

    // Pass 2: Compute forces
    {
      const pass = commandEncoder.beginComputePass({ label: 'compute-forces' })
      pass.setPipeline(this.forcePipeline)
      pass.setBindGroup(0, this.bindGroup)
      pass.dispatchWorkgroups(workgroupCount(this.numAtoms, 64))
      pass.end()
    }

    this.ctx.submit([commandEncoder.finish()])

    return {}
  }

  /**
   * Compute forces and return potential energy (async)
   */
  async computeWithEnergy(
    state: SimulationState,
    neighborList: NeighborList
  ): Promise<number> {
    this.compute(state, neighborList, true)

    // Read back energy
    const commandEncoder = this.ctx.createCommandEncoder()
    commandEncoder.copyBufferToBuffer(
      this.energyBuffer,
      0,
      this.energyStagingBuffer,
      0,
      this.numAtoms * 4
    )
    this.ctx.submit([commandEncoder.finish()])

    await this.energyStagingBuffer.mapAsync(GPUMapMode.READ)
    const data = new Float32Array(this.energyStagingBuffer.getMappedRange())
    
    // Sum up per-atom energies
    let totalEnergy = 0
    for (let i = 0; i < this.numAtoms; i++) {
      totalEnergy += data[i]
    }
    
    this.energyStagingBuffer.unmap()
    return totalEnergy
  }

  /**
   * Update parameter buffer
   */
  private updateParams(maxNeighbors: number, computeEnergy: boolean): void {
    const buffer = new ArrayBuffer(48)
    const u32View = new Uint32Array(buffer)
    const f32View = new Float32Array(buffer)

    // vec4: numAtoms, numTypes, maxNeighbors, computeEnergy
    u32View[0] = this.numAtoms
    u32View[1] = this.numTypes
    u32View[2] = maxNeighbors
    u32View[3] = computeEnergy ? 1 : 0

    // vec4: boxLx, boxLy, boxLz, periodicX
    f32View[4] = this.boxLx
    f32View[5] = this.boxLy
    f32View[6] = this.boxLz
    u32View[7] = this.periodic[0] ? 1 : 0

    // vec4: periodicY, periodicZ, padding, padding
    u32View[8] = this.periodic[1] ? 1 : 0
    u32View[9] = this.periodic[2] ? 1 : 0
    u32View[10] = 0
    u32View[11] = 0

    this.ctx.writeBuffer(this.paramsBuffer, new Float32Array(buffer))
  }

  /**
   * Get bind group layout
   */
  getBindGroupLayout(): GPUBindGroupLayout {
    return this.bindGroupLayout
  }

  /**
   * Destroy GPU resources
   */
  destroy(): void {
    super.destroy()
    this.paramsBuffer.destroy()
    this.energyBuffer.destroy()
    this.energyStagingBuffer.destroy()
  }
}

