// Test setup file for vitest
// Mock browser APIs not available in Node.js

// Mock WebGPU - we can't actually run GPU code in tests
// but we can verify TypeScript logic
Object.defineProperty(globalThis, 'navigator', {
  value: {
    gpu: undefined, // WebGPU not available in tests
  },
  writable: true,
})

