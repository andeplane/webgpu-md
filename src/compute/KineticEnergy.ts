import { WebGPUContext, uniformBufferEntry, storageBufferEntry, bufferEntry, workgroupCount } from './WebGPUContext'
import type { SimulationState } from '../core/SimulationState'
import kineticEnergyShader from '../shaders/kineticEnergy.wgsl?raw'

/**
 * GPU-based kinetic energy computation
 * Uses parallel reduction to compute KE = 0.5 * sum(m_i * v_i^2)
 */
export class KineticEnergy {
  private ctx: WebGPUContext
  private numAtoms: number
  
  // GPU buffers
  private paramsBuffer: GPUBuffer
  private perAtomKEBuffer: GPUBuffer
  private partialSumsBuffer: GPUBuffer
  private resultBuffer: GPUBuffer
  private stagingBuffer: GPUBuffer
  
  // Pipelines
  private perAtomPipeline: GPUComputePipeline
  private reducePipeline: GPUComputePipeline
  private bindGroupLayout: GPUBindGroupLayout
  private bindGroup: GPUBindGroup | null = null
  
  // Workgroup configuration
  private readonly workgroupSize = 256
  private numWorkgroups: number
  
  constructor(ctx: WebGPUContext, numAtoms: number) {
    this.ctx = ctx
    this.numAtoms = numAtoms
    this.numWorkgroups = Math.ceil(numAtoms / this.workgroupSize)
    
    // Create parameter buffer (16 bytes for alignment)
    this.paramsBuffer = ctx.createUniformBuffer(16, 'ke-params')
    
    // Per-atom kinetic energy buffer
    this.perAtomKEBuffer = ctx.createStorageBuffer(
      numAtoms * 4,
      'per-atom-ke'
    )
    
    // Partial sums from each workgroup
    this.partialSumsBuffer = ctx.createStorageBuffer(
      this.numWorkgroups * 4,
      'ke-partial-sums'
    )
    
    // Final result buffer
    this.resultBuffer = ctx.createStorageBuffer(
      4,
      'ke-result',
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    )
    
    // Staging buffer for CPU readback
    this.stagingBuffer = ctx.device.createBuffer({
      size: this.numWorkgroups * 4,  // Read all partial sums
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'ke-staging',
    })
    
    // Create bind group layout
    this.bindGroupLayout = ctx.createBindGroupLayout([
      uniformBufferEntry(0),           // params
      storageBufferEntry(1, true),     // velocities (read-only)
      storageBufferEntry(2, true),     // masses (read-only)
      storageBufferEntry(3, false),    // perAtomKE (read-write)
      storageBufferEntry(4, false),    // result (read-write)
    ], 'ke-layout')
    
    // Create pipelines
    this.perAtomPipeline = ctx.createComputePipeline(
      kineticEnergyShader,
      'computePerAtomKE',
      [this.bindGroupLayout],
      'ke-per-atom'
    )
    
    this.reducePipeline = ctx.createComputePipeline(
      kineticEnergyShader,
      'reduceKE',
      [this.bindGroupLayout],
      'ke-reduce'
    )
    
    // Initialize params
    this.updateParams()
  }
  
  private updateParams(): void {
    const buffer = new ArrayBuffer(16)
    const u32View = new Uint32Array(buffer)
    u32View[0] = this.numAtoms
    u32View[1] = 0
    u32View[2] = 0
    u32View[3] = 0
    this.ctx.writeBuffer(this.paramsBuffer, new Float32Array(buffer))
  }
  
  /**
   * Compute kinetic energy on GPU
   * Returns: { kineticEnergy, temperature } in LJ reduced units
   */
  async compute(state: SimulationState): Promise<{ kineticEnergy: number; temperature: number }> {
    // Create bind group if needed
    if (!this.bindGroup) {
      this.bindGroup = this.ctx.createBindGroup(
        this.bindGroupLayout,
        [
          bufferEntry(0, this.paramsBuffer),
          bufferEntry(1, state.velocitiesBuffer),
          bufferEntry(2, state.massesBuffer),
          bufferEntry(3, this.perAtomKEBuffer),
          bufferEntry(4, this.partialSumsBuffer),
        ],
        'ke-bind-group'
      )
    }
    
    const commandEncoder = this.ctx.createCommandEncoder('ke-compute')
    
    // Phase 1: Compute per-atom KE
    {
      const pass = commandEncoder.beginComputePass({ label: 'per-atom-ke' })
      pass.setPipeline(this.perAtomPipeline)
      pass.setBindGroup(0, this.bindGroup)
      pass.dispatchWorkgroups(workgroupCount(this.numAtoms, this.workgroupSize))
      pass.end()
    }
    
    // Phase 2: Parallel reduction
    {
      const pass = commandEncoder.beginComputePass({ label: 'reduce-ke' })
      pass.setPipeline(this.reducePipeline)
      pass.setBindGroup(0, this.bindGroup)
      pass.dispatchWorkgroups(this.numWorkgroups)
      pass.end()
    }
    
    // Copy partial sums to staging buffer
    commandEncoder.copyBufferToBuffer(
      this.partialSumsBuffer,
      0,
      this.stagingBuffer,
      0,
      this.numWorkgroups * 4
    )
    
    this.ctx.submit([commandEncoder.finish()])
    
    // Wait and read back
    await this.ctx.waitForGPU()
    await this.stagingBuffer.mapAsync(GPUMapMode.READ)
    const data = new Float32Array(this.stagingBuffer.getMappedRange())
    
    // Sum partial results (small final sum on CPU - just numWorkgroups values)
    let totalKE = 0
    for (let i = 0; i < this.numWorkgroups; i++) {
      totalKE += data[i]
    }
    
    this.stagingBuffer.unmap()
    
    // Temperature in LJ reduced units: T* = (2/3) * KE / N
    // (kB = 1 in reduced units)
    const temperature = (2.0 / 3.0) * totalKE / this.numAtoms
    
    return { kineticEnergy: totalKE, temperature }
  }
  
  destroy(): void {
    this.paramsBuffer.destroy()
    this.perAtomKEBuffer.destroy()
    this.partialSumsBuffer.destroy()
    this.resultBuffer.destroy()
    this.stagingBuffer.destroy()
  }
}

