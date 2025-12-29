import { WebGPUContext, storageBufferEntry, uniformBufferEntry, bufferEntry, workgroupCount } from './WebGPUContext'
import { SimulationBox } from '../core/SimulationBox'
import cellListShader from '../shaders/cellList.wgsl?raw'

/**
 * Cell list parameters (matches WGSL struct)
 */
interface CellListParams {
  numAtoms: number
  numCellsX: number
  numCellsY: number
  numCellsZ: number
  cellSizeX: number
  cellSizeY: number
  cellSizeZ: number
  maxAtomsPerCell: number
  originX: number
  originY: number
  originZ: number
  boxLx: number
  boxLy: number
  boxLz: number
}

/**
 * Manages cell list construction on GPU
 * Bins atoms into spatial cells for efficient neighbor searching
 */
export class CellList {
  private ctx: WebGPUContext
  private numAtoms: number
  private cutoff: number
  private maxAtomsPerCell: number

  // Cell grid dimensions
  numCellsX = 0
  numCellsY = 0
  numCellsZ = 0
  numCells = 0
  cellSizeX = 0
  cellSizeY = 0
  cellSizeZ = 0

  // GPU buffers
  paramsBuffer: GPUBuffer
  cellCountsBuffer: GPUBuffer
  cellAtomsBuffer: GPUBuffer
  atomCellBuffer: GPUBuffer

  // Pipelines
  private resetPipeline: GPUComputePipeline
  private binPipeline: GPUComputePipeline
  private bindGroupLayout: GPUBindGroupLayout
  private bindGroup: GPUBindGroup | null = null

  constructor(
    ctx: WebGPUContext,
    numAtoms: number,
    cutoff: number,
    maxAtomsPerCell = 64
  ) {
    this.ctx = ctx
    this.numAtoms = numAtoms
    this.cutoff = cutoff
    this.maxAtomsPerCell = maxAtomsPerCell

    // Create parameter buffer (will be updated when box changes)
    // Size: 16 floats = 64 bytes (aligned to 16)
    this.paramsBuffer = ctx.createUniformBuffer(64, 'cell-list-params')

    // Placeholder buffers - will be resized when box is set
    this.cellCountsBuffer = ctx.createStorageBuffer(4, 'cell-counts')
    this.cellAtomsBuffer = ctx.createStorageBuffer(4, 'cell-atoms')
    this.atomCellBuffer = ctx.createStorageBuffer(numAtoms * 4, 'atom-cell')

    // Create bind group layout
    this.bindGroupLayout = ctx.createBindGroupLayout([
      uniformBufferEntry(0),           // params
      storageBufferEntry(1, true),     // positions (read-only)
      storageBufferEntry(2, false),    // cellCounts (read-write)
      storageBufferEntry(3, false),    // cellAtoms (read-write)
      storageBufferEntry(4, false),    // atomCell (read-write)
    ], 'cell-list-layout')

    // Create pipelines
    this.resetPipeline = ctx.createComputePipeline(
      cellListShader,
      'resetCells',
      [this.bindGroupLayout],
      'cell-list-reset'
    )

    this.binPipeline = ctx.createComputePipeline(
      cellListShader,
      'binAtoms',
      [this.bindGroupLayout],
      'cell-list-bin'
    )
  }

  /**
   * Update cell list for a new box size
   */
  updateBox(box: SimulationBox): void {
    const [lx, ly, lz] = box.dimensions

    // Cell size should be at least cutoff (LAMMPS uses cutoff/2 for optimal binning)
    // Using cutoff directly for simplicity
    this.cellSizeX = this.cutoff
    this.cellSizeY = this.cutoff
    this.cellSizeZ = this.cutoff

    // Compute number of cells in each dimension
    this.numCellsX = Math.max(1, Math.floor(lx / this.cellSizeX))
    this.numCellsY = Math.max(1, Math.floor(ly / this.cellSizeY))
    this.numCellsZ = Math.max(1, Math.floor(lz / this.cellSizeZ))
    this.numCells = this.numCellsX * this.numCellsY * this.numCellsZ

    // Adjust cell size to fit box exactly
    this.cellSizeX = lx / this.numCellsX
    this.cellSizeY = ly / this.numCellsY
    this.cellSizeZ = lz / this.numCellsZ

    console.log(`Cell list: ${this.numCellsX}x${this.numCellsY}x${this.numCellsZ} = ${this.numCells} cells`)
    console.log(`Cell size: ${this.cellSizeX.toFixed(3)} x ${this.cellSizeY.toFixed(3)} x ${this.cellSizeZ.toFixed(3)}`)

    // Update parameter buffer
    const params = this.packParams({
      numAtoms: this.numAtoms,
      numCellsX: this.numCellsX,
      numCellsY: this.numCellsY,
      numCellsZ: this.numCellsZ,
      cellSizeX: this.cellSizeX,
      cellSizeY: this.cellSizeY,
      cellSizeZ: this.cellSizeZ,
      maxAtomsPerCell: this.maxAtomsPerCell,
      originX: box.origin.x,
      originY: box.origin.y,
      originZ: box.origin.z,
      boxLx: lx,
      boxLy: ly,
      boxLz: lz,
    })
    this.ctx.writeBuffer(this.paramsBuffer, params)

    // Resize cell buffers
    this.cellCountsBuffer.destroy()
    this.cellAtomsBuffer.destroy()

    this.cellCountsBuffer = this.ctx.createStorageBuffer(
      this.numCells * 4,
      'cell-counts'
    )

    this.cellAtomsBuffer = this.ctx.createStorageBuffer(
      this.numCells * this.maxAtomsPerCell * 4,
      'cell-atoms'
    )

    // Invalidate bind group (will be recreated on next build)
    this.bindGroup = null
  }

  /**
   * Build the cell list for given positions buffer
   */
  build(positionsBuffer: GPUBuffer): void {
    // Create bind group if needed
    if (!this.bindGroup) {
      this.bindGroup = this.ctx.createBindGroup(
        this.bindGroupLayout,
        [
          bufferEntry(0, this.paramsBuffer),
          bufferEntry(1, positionsBuffer),
          bufferEntry(2, this.cellCountsBuffer),
          bufferEntry(3, this.cellAtomsBuffer),
          bufferEntry(4, this.atomCellBuffer),
        ],
        'cell-list-bind-group'
      )
    }

    const commandEncoder = this.ctx.createCommandEncoder('cell-list')

    // Pass 1: Reset cell counts
    {
      const pass = commandEncoder.beginComputePass({ label: 'reset-cells' })
      pass.setPipeline(this.resetPipeline)
      pass.setBindGroup(0, this.bindGroup)
      pass.dispatchWorkgroups(workgroupCount(this.numCells, 256))
      pass.end()
    }

    // Pass 2: Bin atoms into cells
    {
      const pass = commandEncoder.beginComputePass({ label: 'bin-atoms' })
      pass.setPipeline(this.binPipeline)
      pass.setBindGroup(0, this.bindGroup)
      pass.dispatchWorkgroups(workgroupCount(this.numAtoms, 256))
      pass.end()
    }

    this.ctx.submit([commandEncoder.finish()])
  }

  /**
   * Pack parameters into a Float32Array for GPU upload
   */
  private packParams(p: CellListParams): Float32Array {
    return new Float32Array([
      // First vec4: numAtoms, numCellsX, numCellsY, numCellsZ (as u32, but stored as f32 bits)
      p.numAtoms,
      p.numCellsX,
      p.numCellsY,
      p.numCellsZ,
      // Second vec4: cellSizeX, cellSizeY, cellSizeZ, maxAtomsPerCell
      p.cellSizeX,
      p.cellSizeY,
      p.cellSizeZ,
      p.maxAtomsPerCell,
      // Third vec4: originX, originY, originZ, boxLx
      p.originX,
      p.originY,
      p.originZ,
      p.boxLx,
      // Fourth vec4: boxLy, boxLz, padding, padding
      p.boxLy,
      p.boxLz,
      0,
      0,
    ])
  }

  /**
   * Get the stencil offsets for 27 neighboring cells (3x3x3)
   * Returns array of [dx, dy, dz] offsets
   */
  getStencilOffsets(): Int32Array {
    const offsets: number[] = []
    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          offsets.push(dx, dy, dz)
        }
      }
    }
    return new Int32Array(offsets)
  }

  /**
   * Destroy GPU resources
   */
  destroy(): void {
    this.paramsBuffer.destroy()
    this.cellCountsBuffer.destroy()
    this.cellAtomsBuffer.destroy()
    this.atomCellBuffer.destroy()
  }
}

