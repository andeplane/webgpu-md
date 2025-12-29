// Test setup file for vitest
// Provides WebGPU support in Node.js using Dawn bindings

import { create, globals } from 'webgpu'

// Assign WebGPU constants (GPUBufferUsage, etc.) to global scope
Object.assign(globalThis, globals)

// Create navigator.gpu using Dawn implementation
const gpu = create([])

Object.defineProperty(globalThis, 'navigator', {
  value: { gpu },
  writable: true,
  configurable: true,
})

// Export a helper to get WebGPU adapter and device for tests
export async function getWebGPUDevice(): Promise<GPUDevice | null> {
  try {
    const adapter = await navigator.gpu?.requestAdapter()
    if (!adapter) return null
    return await adapter.requestDevice()
  } catch {
    return null
  }
}

