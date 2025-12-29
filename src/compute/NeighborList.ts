import { WebGPUContext, storageBufferEntry, uniformBufferEntry, bufferEntry, workgroupCount } from './WebGPUContext'
import { CellList } from './CellList'
import { SimulationBox } from '../core/SimulationBox'
import neighborListShader from '../shaders/neighborList.wgsl?raw'

/**
 * Neighbor list parameters (matches WGSL struct)
 */
interface NeighborListParams {
  numAtoms: number
  numCellsX: number
  numCellsY: number
  numCellsZ: number
  maxNeighbors: number
  cutoffSq: number
  maxAtomsPerCell: number
  originX: number
  originY: number
  originZ: number
  boxLx: number
  boxLy: number
  boxLz: number
  periodicX: number
  periodicY: number
  periodicZ: number
}

/**
 * Manages neighbor list construction on GPU
 * Uses cell lists for O(N) neighbor searching
 */
export class NeighborList {
  private ctx: WebGPUContext
  private numAtoms: number
  private cutoff: number
  private skin: number
  readonly maxNeighbors: number

  // Associated cell list
  readonly cellList: CellList

  // GPU buffers
  paramsBuffer: GPUBuffer
  neighborListBuffer: GPUBuffer  // [numAtoms * maxNeighbors] - flat array of neighbor indices
  numNeighborsBuffer: GPUBuffer  // [numAtoms] - count of neighbors per atom

  // Pipelines
  private buildPipeline: GPUComputePipeline
  private bindGroupLayout: GPUBindGroupLayout
  private bindGroup: GPUBindGroup | null = null

  // Current box (for rebuild tracking)
  private currentBox: SimulationBox | null = null

  constructor(
    ctx: WebGPUContext,
    numAtoms: number,
    cutoff: number,
    skin = 0.3,
    maxNeighbors = 128,
    maxAtomsPerCell = 64
  ) {
    this.ctx = ctx
    this.numAtoms = numAtoms
    this.cutoff = cutoff
    this.skin = skin
    this.maxNeighbors = maxNeighbors

    // Create cell list with cutoff + skin as the cell size
    // This ensures all potential neighbors are in adjacent cells
    this.cellList = new CellList(ctx, numAtoms, cutoff + skin, maxAtomsPerCell)

    // Create parameter buffer
    // Struct size: 6 vec4 = 24 floats = 96 bytes
    this.paramsBuffer = ctx.createUniformBuffer(96, 'neighbor-list-params')

    // Create neighbor list buffers
    this.neighborListBuffer = ctx.createStorageBuffer(
      numAtoms * maxNeighbors * 4,
      'neighbor-list'
    )

    this.numNeighborsBuffer = ctx.createStorageBuffer(
      numAtoms * 4,
      'num-neighbors'
    )

    // Create bind group layout
    this.bindGroupLayout = ctx.createBindGroupLayout([
      uniformBufferEntry(0),           // params
      storageBufferEntry(1, true),     // positions (read-only)
      storageBufferEntry(2, true),     // cellCounts (read-only)
      storageBufferEntry(3, true),     // cellAtoms (read-only)
      storageBufferEntry(4, true),     // atomCell (read-only)
      storageBufferEntry(5, false),    // neighborList (read-write)
      storageBufferEntry(6, false),    // numNeighbors (read-write)
    ], 'neighbor-list-layout')

    // Create pipeline
    this.buildPipeline = ctx.createComputePipeline(
      neighborListShader,
      'buildNeighborList',
      [this.bindGroupLayout],
      'neighbor-list-build'
    )
  }

  /**
   * Get the effective cutoff (cutoff + skin)
   */
  get effectiveCutoff(): number {
    return this.cutoff + this.skin
  }

  /**
   * Update for a new simulation box
   */
  updateBox(box: SimulationBox): void {
    this.currentBox = box
    this.cellList.updateBox(box)
    this.updateParams(box)
    this.bindGroup = null  // Force bind group recreation
  }

  /**
   * Update parameter buffer
   */
  private updateParams(box: SimulationBox): void {
    const [lx, ly, lz] = box.dimensions
    const effectiveCutoffSq = this.effectiveCutoff * this.effectiveCutoff

    const params = this.packParams({
      numAtoms: this.numAtoms,
      numCellsX: this.cellList.numCellsX,
      numCellsY: this.cellList.numCellsY,
      numCellsZ: this.cellList.numCellsZ,
      maxNeighbors: this.maxNeighbors,
      cutoffSq: effectiveCutoffSq,
      maxAtomsPerCell: this.cellList.numCells > 0 ? 64 : 0,  // Match cell list
      originX: box.origin.x,
      originY: box.origin.y,
      originZ: box.origin.z,
      boxLx: lx,
      boxLy: ly,
      boxLz: lz,
      periodicX: box.periodic[0] ? 1 : 0,
      periodicY: box.periodic[1] ? 1 : 0,
      periodicZ: box.periodic[2] ? 1 : 0,
    })

    this.ctx.writeBuffer(this.paramsBuffer, params)
  }

  /**
   * Build the neighbor list for given positions
   * This also rebuilds the cell list
   */
  build(positionsBuffer: GPUBuffer): void {
    if (!this.currentBox) {
      throw new Error('Must call updateBox before building neighbor list')
    }

    // First, build the cell list
    this.cellList.build(positionsBuffer)

    // Create bind group if needed
    if (!this.bindGroup) {
      this.bindGroup = this.ctx.createBindGroup(
        this.bindGroupLayout,
        [
          bufferEntry(0, this.paramsBuffer),
          bufferEntry(1, positionsBuffer),
          bufferEntry(2, this.cellList.cellCountsBuffer),
          bufferEntry(3, this.cellList.cellAtomsBuffer),
          bufferEntry(4, this.cellList.atomCellBuffer),
          bufferEntry(5, this.neighborListBuffer),
          bufferEntry(6, this.numNeighborsBuffer),
        ],
        'neighbor-list-bind-group'
      )
    }

    // Build neighbor list
    this.ctx.runComputePass(
      this.buildPipeline,
      [this.bindGroup],
      workgroupCount(this.numAtoms, 64),
      1,
      1,
      'build-neighbor-list'
    )
  }

  /**
   * Pack parameters into a Float32Array for GPU upload
   */
  private packParams(p: NeighborListParams): Float32Array {
    // Using Uint32Array for integer values, then copy to Float32Array
    const buffer = new ArrayBuffer(96)
    const u32View = new Uint32Array(buffer)
    const f32View = new Float32Array(buffer)

    // First vec4: numAtoms, numCellsX, numCellsY, numCellsZ (u32)
    u32View[0] = p.numAtoms
    u32View[1] = p.numCellsX
    u32View[2] = p.numCellsY
    u32View[3] = p.numCellsZ

    // Second vec4: maxNeighbors, cutoffSq, maxAtomsPerCell, padding (mixed)
    u32View[4] = p.maxNeighbors
    f32View[5] = p.cutoffSq
    u32View[6] = p.maxAtomsPerCell
    u32View[7] = 0  // padding

    // Third vec4: originX, originY, originZ, boxLx (f32)
    f32View[8] = p.originX
    f32View[9] = p.originY
    f32View[10] = p.originZ
    f32View[11] = p.boxLx

    // Fourth vec4: boxLy, boxLz, periodicX, periodicY (mixed)
    f32View[12] = p.boxLy
    f32View[13] = p.boxLz
    u32View[14] = p.periodicX
    u32View[15] = p.periodicY

    // Fifth vec4: periodicZ, padding, padding, padding (u32)
    u32View[16] = p.periodicZ
    u32View[17] = 0
    u32View[18] = 0
    u32View[19] = 0

    // Sixth vec4: padding
    u32View[20] = 0
    u32View[21] = 0
    u32View[22] = 0
    u32View[23] = 0

    return new Float32Array(buffer)
  }

  /**
   * Read neighbor counts back from GPU for debugging
   */
  async readNeighborCounts(): Promise<Uint32Array> {
    const stagingBuffer = this.ctx.device.createBuffer({
      size: this.numAtoms * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })
    
    const commandEncoder = this.ctx.createCommandEncoder()
    commandEncoder.copyBufferToBuffer(
      this.numNeighborsBuffer,
      0,
      stagingBuffer,
      0,
      this.numAtoms * 4
    )
    this.ctx.submit([commandEncoder.finish()])
    
    await this.ctx.waitForGPU()
    await stagingBuffer.mapAsync(GPUMapMode.READ)
    const data = new Uint32Array(stagingBuffer.getMappedRange().slice(0))
    stagingBuffer.unmap()
    stagingBuffer.destroy()
    
    return data
  }

  /**
   * Destroy GPU resources
   */
  destroy(): void {
    this.paramsBuffer.destroy()
    this.neighborListBuffer.destroy()
    this.numNeighborsBuffer.destroy()
    this.cellList.destroy()
  }
}

