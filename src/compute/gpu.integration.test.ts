import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { getWebGPUDevice } from '../setupTests'

/**
 * GPU Integration Tests
 * 
 * These tests run actual WebGPU compute shaders using Dawn bindings.
 * They verify that:
 * - Shaders compile successfully
 * - Compute pipelines execute correctly
 * - Buffer data is computed as expected
 */

describe('GPU Integration Tests', () => {
  let device: GPUDevice | null = null

  beforeAll(async () => {
    device = await getWebGPUDevice()
  })

  afterAll(() => {
    device?.destroy()
  })

  describe('WebGPU Device', () => {
    it('should successfully request a WebGPU device', () => {
      expect(device).not.toBeNull()
    })

    it('should have expected device limits', () => {
      if (!device) return

      expect(device.limits.maxComputeWorkgroupSizeX).toBeGreaterThanOrEqual(64)
      expect(device.limits.maxStorageBufferBindingSize).toBeGreaterThan(0)
    })
  })

  describe('Basic Compute Shader', () => {
    it('should compile and execute a simple compute shader', async () => {
      if (!device) return

      // Simple shader that doubles values in a buffer
      const shaderCode = `
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let idx = id.x;
          if (idx < arrayLength(&data)) {
            data[idx] = data[idx] * 2.0;
          }
        }
      `

      // Create shader module
      const shaderModule = device.createShaderModule({ code: shaderCode })

      // Create compute pipeline
      const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'main',
        },
      })

      // Create input data
      const inputData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8])
      const bufferSize = inputData.byteLength

      // Create GPU buffer
      const gpuBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      })

      // Upload data
      device.queue.writeBuffer(gpuBuffer, 0, inputData)

      // Create bind group
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: gpuBuffer } }],
      })

      // Run compute shader
      const commandEncoder = device.createCommandEncoder()
      const passEncoder = commandEncoder.beginComputePass()
      passEncoder.setPipeline(pipeline)
      passEncoder.setBindGroup(0, bindGroup)
      passEncoder.dispatchWorkgroups(Math.ceil(inputData.length / 64))
      passEncoder.end()

      // Copy result to staging buffer
      const stagingBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })
      commandEncoder.copyBufferToBuffer(gpuBuffer, 0, stagingBuffer, 0, bufferSize)

      device.queue.submit([commandEncoder.finish()])

      // Read result
      await stagingBuffer.mapAsync(GPUMapMode.READ)
      const result = new Float32Array(stagingBuffer.getMappedRange().slice(0))
      stagingBuffer.unmap()

      // Verify values doubled
      expect(result[0]).toBe(2)
      expect(result[1]).toBe(4)
      expect(result[2]).toBe(6)
      expect(result[7]).toBe(16)

      // Cleanup
      gpuBuffer.destroy()
      stagingBuffer.destroy()
    })
  })

  describe('Force Calculation Shader', () => {
    it('should compute LJ forces between two atoms', async () => {
      if (!device) return

      // Simplified LJ force shader for testing
      const shaderCode = `
        struct Params {
          numAtoms: u32,
          epsilon: f32,
          sigma: f32,
          cutoffSq: f32,
        }

        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read> positions: array<f32>;
        @group(0) @binding(2) var<storage, read_write> forces: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let i = id.x;
          if (i >= params.numAtoms) { return; }

          // Atom i position
          let xi = positions[i * 3u];
          let yi = positions[i * 3u + 1u];
          let zi = positions[i * 3u + 2u];

          var fx: f32 = 0.0;
          var fy: f32 = 0.0;
          var fz: f32 = 0.0;

          // Simple loop over all other atoms (no neighbor list)
          for (var j: u32 = 0u; j < params.numAtoms; j++) {
            if (j == i) { continue; }

            let xj = positions[j * 3u];
            let yj = positions[j * 3u + 1u];
            let zj = positions[j * 3u + 2u];

            let dx = xi - xj;
            let dy = yi - yj;
            let dz = zi - zj;
            let rsq = dx * dx + dy * dy + dz * dz;

            if (rsq < params.cutoffSq && rsq > 0.0) {
              let r2inv = 1.0 / rsq;
              let r6inv = r2inv * r2inv * r2inv;
              let sigma6 = params.sigma * params.sigma * params.sigma * 
                          params.sigma * params.sigma * params.sigma;
              
              // F = 48 * epsilon * sigma^12 / r^13 - 24 * epsilon * sigma^6 / r^7
              // F/r = 48 * epsilon * sigma^12 / r^14 - 24 * epsilon * sigma^6 / r^8
              //     = r2inv * r6inv * (48 * epsilon * sigma6 * sigma6 * r6inv - 24 * epsilon * sigma6)
              let forcelj = r6inv * (48.0 * params.epsilon * sigma6 * sigma6 * r6inv - 
                                      24.0 * params.epsilon * sigma6);
              let fpair = forcelj * r2inv;

              fx += dx * fpair;
              fy += dy * fpair;
              fz += dz * fpair;
            }
          }

          forces[i * 3u] = fx;
          forces[i * 3u + 1u] = fy;
          forces[i * 3u + 2u] = fz;
        }
      `

      const shaderModule = device.createShaderModule({ code: shaderCode })

      // Parameters: 2 atoms, epsilon=1.0, sigma=1.0, cutoff=2.5
      const paramsData = new Float32Array([
        2,    // numAtoms (as float, will be read as u32)
        1.0,  // epsilon
        1.0,  // sigma
        6.25, // cutoffSq = 2.5^2
      ])
      // Fix: numAtoms should be u32
      const paramsBuffer = device.createBuffer({
        size: 16, // 4 values * 4 bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      const paramsU32 = new Uint32Array(paramsData.buffer)
      paramsU32[0] = 2 // numAtoms as u32
      device.queue.writeBuffer(paramsBuffer, 0, paramsData)

      // Two atoms at distance sigma (r = 1.0)
      // At r = sigma, LJ force = 0 (equilibrium)
      // Let's use r = 2^(1/6) * sigma ≈ 1.122 for zero force
      // Or use r = 0.9 sigma for repulsive force
      const positions = new Float32Array([
        0.0, 0.0, 0.0,  // Atom 0 at origin
        0.9, 0.0, 0.0,  // Atom 1 at x=0.9 (within sigma, repulsive)
      ])
      const positionsBuffer = device.createBuffer({
        size: positions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(positionsBuffer, 0, positions)

      // Forces buffer
      const forcesBuffer = device.createBuffer({
        size: positions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      })

      // Create bind group layout
      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      })

      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      })

      const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: 'main' },
      })

      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: positionsBuffer } },
          { binding: 2, resource: { buffer: forcesBuffer } },
        ],
      })

      // Execute
      const commandEncoder = device.createCommandEncoder()
      const pass = commandEncoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(1)
      pass.end()

      // Read back forces
      const stagingBuffer = device.createBuffer({
        size: positions.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })
      commandEncoder.copyBufferToBuffer(forcesBuffer, 0, stagingBuffer, 0, positions.byteLength)
      device.queue.submit([commandEncoder.finish()])

      await stagingBuffer.mapAsync(GPUMapMode.READ)
      const forces = new Float32Array(stagingBuffer.getMappedRange().slice(0))
      stagingBuffer.unmap()

      // At r = 0.9, atoms should repel (positive force on atom 0 in -x direction)
      // Force on atom 0 should be negative x (pushed away from atom 1)
      // Force on atom 1 should be positive x (pushed away from atom 0)
      expect(forces[0]).toBeLessThan(0) // Atom 0 pushed in -x
      expect(forces[3]).toBeGreaterThan(0) // Atom 1 pushed in +x
      
      // Forces should be equal and opposite (Newton's 3rd law)
      expect(forces[0]).toBeCloseTo(-forces[3], 4)
      
      // Y and Z forces should be zero
      expect(forces[1]).toBeCloseTo(0, 6)
      expect(forces[2]).toBeCloseTo(0, 6)

      // Cleanup
      paramsBuffer.destroy()
      positionsBuffer.destroy()
      forcesBuffer.destroy()
      stagingBuffer.destroy()
    })
  })

  describe('Velocity Verlet Integration', () => {
    it('should integrate position and velocity correctly', async () => {
      if (!device) return

      // Simple velocity verlet integration shader
      const shaderCode = `
        struct Params {
          numAtoms: u32,
          dt: f32,
          dtSqHalf: f32,
          dtHalf: f32,
        }

        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read_write> positions: array<f32>;
        @group(0) @binding(2) var<storage, read_write> velocities: array<f32>;
        @group(0) @binding(3) var<storage, read> forces: array<f32>;
        @group(0) @binding(4) var<storage, read> masses: array<f32>;

        @compute @workgroup_size(64)
        fn integrateInitial(@builtin(global_invocation_id) id: vec3<u32>) {
          let i = id.x;
          if (i >= params.numAtoms) { return; }

          let mass = masses[i];
          let invMass = 1.0 / mass;

          // Update velocity: v += 0.5 * dt * f / m
          velocities[i * 3u] += params.dtHalf * forces[i * 3u] * invMass;
          velocities[i * 3u + 1u] += params.dtHalf * forces[i * 3u + 1u] * invMass;
          velocities[i * 3u + 2u] += params.dtHalf * forces[i * 3u + 2u] * invMass;

          // Update position: x += v * dt
          positions[i * 3u] += velocities[i * 3u] * params.dt;
          positions[i * 3u + 1u] += velocities[i * 3u + 1u] * params.dt;
          positions[i * 3u + 2u] += velocities[i * 3u + 2u] * params.dt;
        }
      `

      const shaderModule = device.createShaderModule({ code: shaderCode })

      // Setup: 1 atom, dt=0.001
      const dt = 0.001
      const paramsData = new ArrayBuffer(16)
      new Uint32Array(paramsData, 0, 1)[0] = 1 // numAtoms
      new Float32Array(paramsData, 4, 3).set([dt, 0.5 * dt * dt, 0.5 * dt])

      const paramsBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(paramsBuffer, 0, paramsData)

      // Initial position at origin, velocity = (1, 0, 0)
      const positions = new Float32Array([0, 0, 0])
      const velocities = new Float32Array([1, 0, 0])
      const forces = new Float32Array([0, 0, 0]) // No force
      const masses = new Float32Array([1.0])

      const posBuffer = device.createBuffer({
        size: 12,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      })
      const velBuffer = device.createBuffer({
        size: 12,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      })
      const forceBuffer = device.createBuffer({
        size: 12,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      const massBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })

      device.queue.writeBuffer(posBuffer, 0, positions)
      device.queue.writeBuffer(velBuffer, 0, velocities)
      device.queue.writeBuffer(forceBuffer, 0, forces)
      device.queue.writeBuffer(massBuffer, 0, masses)

      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ],
      })

      const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: 'integrateInitial' },
      })

      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: posBuffer } },
          { binding: 2, resource: { buffer: velBuffer } },
          { binding: 3, resource: { buffer: forceBuffer } },
          { binding: 4, resource: { buffer: massBuffer } },
        ],
      })

      // Execute
      const commandEncoder = device.createCommandEncoder()
      const pass = commandEncoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(1)
      pass.end()

      // Read positions
      const stagingBuffer = device.createBuffer({
        size: 12,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })
      commandEncoder.copyBufferToBuffer(posBuffer, 0, stagingBuffer, 0, 12)
      device.queue.submit([commandEncoder.finish()])

      await stagingBuffer.mapAsync(GPUMapMode.READ)
      const newPositions = new Float32Array(stagingBuffer.getMappedRange().slice(0))
      stagingBuffer.unmap()

      // With v=1, dt=0.001, f=0: x += v * dt = 0 + 1 * 0.001 = 0.001
      expect(newPositions[0]).toBeCloseTo(0.001, 6)
      expect(newPositions[1]).toBeCloseTo(0, 6)
      expect(newPositions[2]).toBeCloseTo(0, 6)

      // Cleanup
      paramsBuffer.destroy()
      posBuffer.destroy()
      velBuffer.destroy()
      forceBuffer.destroy()
      massBuffer.destroy()
      stagingBuffer.destroy()
    })
  })
})

