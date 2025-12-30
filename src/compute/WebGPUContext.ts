/**
 * WebGPU context and compute pipeline utilities
 */

export interface WebGPUContextConfig {
  /** Required features (optional) */
  requiredFeatures?: GPUFeatureName[]
  /** Required limits (optional) */
  requiredLimits?: Record<string, number>
}

/**
 * Manages WebGPU device and provides utilities for compute pipelines
 */
export class WebGPUContext {
  readonly adapter: GPUAdapter
  readonly device: GPUDevice

  private constructor(adapter: GPUAdapter, device: GPUDevice) {
    this.adapter = adapter
    this.device = device
  }

  /**
   * Initialize WebGPU context
   * @throws Error if WebGPU is not supported or device creation fails
   */
  static async create(config: WebGPUContextConfig = {}): Promise<WebGPUContext> {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported in this browser')
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    })

    if (!adapter) {
      throw new Error('Failed to get WebGPU adapter')
    }

    console.log('WebGPU Adapter acquired')

    // Query adapter limits
    const adapterLimits = adapter.limits

    // Request the adapter's maximum storage buffer size
    const requiredLimits: Record<string, number> = {
      ...config.requiredLimits,
      maxStorageBufferBindingSize: adapterLimits.maxStorageBufferBindingSize,
      maxBufferSize: adapterLimits.maxBufferSize,
    }

    const device = await adapter.requestDevice({
      requiredFeatures: config.requiredFeatures,
      requiredLimits,
    })

    // Log limits for debugging
    console.log('WebGPU Device Limits:')
    console.log(`  maxStorageBufferBindingSize: ${(device.limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)} MB`)
    console.log(`  maxComputeWorkgroupStorageSize: ${device.limits.maxComputeWorkgroupStorageSize} bytes`)
    console.log(`  maxComputeInvocationsPerWorkgroup: ${device.limits.maxComputeInvocationsPerWorkgroup}`)

    // Set up error handling
    device.lost.then((info) => {
      if (info.reason !== 'destroyed') {
        console.error('WebGPU device lost:', info.message)
        // Could attempt to recreate device here
        throw new Error(`WebGPU device lost: ${info.message}`)
      }
    })

    device.onuncapturederror = (event) => {
      console.error('WebGPU uncaptured error:', event.error)
    }

    return new WebGPUContext(adapter, device)
  }

  /**
   * Create a compute pipeline from WGSL shader code
   */
  createComputePipeline(
    code: string,
    entryPoint: string,
    bindGroupLayouts: GPUBindGroupLayout[],
    label?: string
  ): GPUComputePipeline {
    const shaderModule = this.device.createShaderModule({
      code,
      label: label ? `${label}-shader` : undefined,
    })

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts,
      label: label ? `${label}-layout` : undefined,
    })

    return this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint,
      },
      label,
    })
  }

  /**
   * Create a bind group layout from entries
   */
  createBindGroupLayout(
    entries: GPUBindGroupLayoutEntry[],
    label?: string
  ): GPUBindGroupLayout {
    return this.device.createBindGroupLayout({
      entries,
      label,
    })
  }

  /**
   * Create a bind group from layout and entries
   */
  createBindGroup(
    layout: GPUBindGroupLayout,
    entries: GPUBindGroupEntry[],
    label?: string
  ): GPUBindGroup {
    return this.device.createBindGroup({
      layout,
      entries,
      label,
    })
  }

  /**
   * Create a storage buffer
   */
  createStorageBuffer(
    size: number,
    label?: string,
    usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  ): GPUBuffer {
    return this.device.createBuffer({
      size,
      usage,
      label,
    })
  }

  /**
   * Create a uniform buffer
   */
  createUniformBuffer(size: number, label?: string): GPUBuffer {
    return this.device.createBuffer({
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label,
    })
  }

  /**
   * Write data to a buffer
   */
  writeBuffer(buffer: GPUBuffer, data: ArrayBufferView, offset = 0): void {
    this.device.queue.writeBuffer(buffer, offset, data.buffer, data.byteOffset, data.byteLength)
  }

  /**
   * Run a compute pass with the given pipeline and bind groups
   */
  runComputePass(
    pipeline: GPUComputePipeline,
    bindGroups: GPUBindGroup[],
    workgroupCountX: number,
    workgroupCountY = 1,
    workgroupCountZ = 1,
    label?: string
  ): void {
    const commandEncoder = this.device.createCommandEncoder({
      label: label ? `${label}-encoder` : undefined,
    })

    const passEncoder = commandEncoder.beginComputePass({
      label: label ? `${label}-pass` : undefined,
    })

    passEncoder.setPipeline(pipeline)
    bindGroups.forEach((bindGroup, index) => {
      passEncoder.setBindGroup(index, bindGroup)
    })
    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ)
    passEncoder.end()

    this.device.queue.submit([commandEncoder.finish()])
  }

  /**
   * Create a command encoder for batching multiple operations
   */
  createCommandEncoder(label?: string): GPUCommandEncoder {
    return this.device.createCommandEncoder({ label })
  }

  /**
   * Submit command buffers to the queue
   */
  submit(commandBuffers: GPUCommandBuffer[]): void {
    this.device.queue.submit(commandBuffers)
  }

  /**
   * Wait for all GPU operations to complete
   */
  async waitForGPU(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone()
  }

  /**
   * Get device limits
   */
  get limits(): GPUSupportedLimits {
    return this.device.limits
  }

  /**
   * Get max workgroup size
   */
  get maxWorkgroupSize(): number {
    return this.device.limits.maxComputeWorkgroupSizeX
  }

  /**
   * Get max workgroups per dimension
   */
  get maxWorkgroupsPerDimension(): number {
    return this.device.limits.maxComputeWorkgroupsPerDimension
  }

  /**
   * Get max storage buffer binding size
   */
  get maxStorageBufferBindingSize(): number {
    return this.device.limits.maxStorageBufferBindingSize
  }

  /**
   * Destroy the device
   */
  destroy(): void {
    this.device.destroy()
  }
}

/**
 * Helper to create bind group layout entries for storage buffers
 */
export function storageBufferEntry(
  binding: number,
  readOnly = false
): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: {
      type: readOnly ? 'read-only-storage' : 'storage',
    },
  }
}

/**
 * Helper to create bind group layout entries for uniform buffers
 */
export function uniformBufferEntry(binding: number): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: {
      type: 'uniform',
    },
  }
}

/**
 * Helper to create bind group entries for buffers
 */
export function bufferEntry(binding: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return {
    binding,
    resource: { buffer },
  }
}

/**
 * Calculate the number of workgroups needed to cover N items
 */
export function workgroupCount(itemCount: number, workgroupSize: number): number {
  return Math.ceil(itemCount / workgroupSize)
}

