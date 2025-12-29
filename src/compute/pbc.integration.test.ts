import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { getWebGPUDevice } from '../setupTests'

/**
 * Periodic Boundary Condition (PBC) and Force Calculation Tests
 * 
 * These tests verify that:
 * - Minimum image convention works correctly
 * - Forces are correct for atoms across periodic boundaries
 * - The LJ potential behaves correctly at various distances
 */

describe('PBC and Force Calculation Tests', () => {
  let device: GPUDevice | null = null

  beforeAll(async () => {
    device = await getWebGPUDevice()
  })

  afterAll(() => {
    device?.destroy()
  })

  describe('Minimum Image Convention', () => {
    it('should compute correct distance for atoms across periodic boundary', async () => {
      if (!device) return

      // Test shader that computes minimum image distance
      const shaderCode = `
        struct Params {
          boxLx: f32,
          boxLy: f32,
          boxLz: f32,
          periodicX: u32,
          periodicY: u32,
          periodicZ: u32,
          _pad1: u32,
          _pad2: u32,
        }

        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read> pos1: vec3<f32>;
        @group(0) @binding(2) var<storage, read> pos2: vec3<f32>;
        @group(0) @binding(3) var<storage, read_write> result: vec3<f32>;
        @group(0) @binding(4) var<storage, read_write> rsq: f32;

        fn minimumImage(dx: f32, dy: f32, dz: f32) -> vec3<f32> {
          var r = vec3<f32>(dx, dy, dz);
          
          if (params.periodicX != 0u) {
            if (r.x > params.boxLx * 0.5) {
              r.x -= params.boxLx;
            } else if (r.x < -params.boxLx * 0.5) {
              r.x += params.boxLx;
            }
          }
          
          if (params.periodicY != 0u) {
            if (r.y > params.boxLy * 0.5) {
              r.y -= params.boxLy;
            } else if (r.y < -params.boxLy * 0.5) {
              r.y += params.boxLy;
            }
          }
          
          if (params.periodicZ != 0u) {
            if (r.z > params.boxLz * 0.5) {
              r.z -= params.boxLz;
            } else if (r.z < -params.boxLz * 0.5) {
              r.z += params.boxLz;
            }
          }
          
          return r;
        }

        @compute @workgroup_size(1)
        fn main() {
          let delta = minimumImage(pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z);
          result = delta;
          rsq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        }
      `

      const shaderModule = device.createShaderModule({ code: shaderCode })

      // Test case: Box 10x10x10, atoms at x=1 and x=9
      // Direct distance = 8, but across PBC = 2
      const paramsData = new ArrayBuffer(32)
      const f32 = new Float32Array(paramsData)
      const u32 = new Uint32Array(paramsData)
      f32[0] = 10.0  // boxLx
      f32[1] = 10.0  // boxLy
      f32[2] = 10.0  // boxLz
      u32[3] = 1     // periodicX
      u32[4] = 1     // periodicY
      u32[5] = 1     // periodicZ

      const paramsBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(paramsBuffer, 0, paramsData)

      // Atom 1 at (1, 0, 0)
      const pos1Buffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(pos1Buffer, 0, new Float32Array([1, 0, 0, 0]))

      // Atom 2 at (9, 0, 0) - across PBC boundary
      const pos2Buffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(pos2Buffer, 0, new Float32Array([9, 0, 0, 0]))

      const resultBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      })

      const rsqBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      })

      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      })

      const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: 'main' },
      })

      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: pos1Buffer } },
          { binding: 2, resource: { buffer: pos2Buffer } },
          { binding: 3, resource: { buffer: resultBuffer } },
          { binding: 4, resource: { buffer: rsqBuffer } },
        ],
      })

      const commandEncoder = device.createCommandEncoder()
      const pass = commandEncoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(1)
      pass.end()

      const resultStaging = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })
      const rsqStaging = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })

      commandEncoder.copyBufferToBuffer(resultBuffer, 0, resultStaging, 0, 16)
      commandEncoder.copyBufferToBuffer(rsqBuffer, 0, rsqStaging, 0, 4)
      device.queue.submit([commandEncoder.finish()])

      await resultStaging.mapAsync(GPUMapMode.READ)
      await rsqStaging.mapAsync(GPUMapMode.READ)

      const resultData = new Float32Array(resultStaging.getMappedRange().slice(0))
      const rsqData = new Float32Array(rsqStaging.getMappedRange().slice(0))

      resultStaging.unmap()
      rsqStaging.unmap()

      // pos1 - pos2 = (1,0,0) - (9,0,0) = (-8,0,0)
      // With PBC on box=10: -8 < -5, so -8 + 10 = 2
      expect(resultData[0]).toBeCloseTo(2, 4)  // dx with MIC
      expect(resultData[1]).toBeCloseTo(0, 6)
      expect(resultData[2]).toBeCloseTo(0, 6)
      expect(rsqData[0]).toBeCloseTo(4, 4)     // r^2 = 2^2 = 4

      // Cleanup
      paramsBuffer.destroy()
      pos1Buffer.destroy()
      pos2Buffer.destroy()
      resultBuffer.destroy()
      rsqBuffer.destroy()
      resultStaging.destroy()
      rsqStaging.destroy()
    })

    it('should NOT wrap when distance is within half box', async () => {
      if (!device) return

      const shaderCode = `
        struct Params {
          boxL: f32,
          periodic: u32,
          _pad1: u32,
          _pad2: u32,
        }

        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read_write> input: f32;
        @group(0) @binding(2) var<storage, read_write> output: f32;

        @compute @workgroup_size(1)
        fn main() {
          var dx = input;
          if (params.periodic != 0u) {
            if (dx > params.boxL * 0.5) {
              dx -= params.boxL;
            } else if (dx < -params.boxL * 0.5) {
              dx += params.boxL;
            }
          }
          output = dx;
        }
      `

      const shaderModule = device.createShaderModule({ code: shaderCode })

      const paramsData = new ArrayBuffer(16)
      new Float32Array(paramsData, 0, 1)[0] = 10.0  // boxL = 10
      new Uint32Array(paramsData, 4, 1)[0] = 1     // periodic = true

      const paramsBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(paramsBuffer, 0, paramsData)

      // Test dx = 3 (within half box = 5, should NOT wrap)
      const inputBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(inputBuffer, 0, new Float32Array([3.0]))

      const outputBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      })

      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      })

      const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: 'main' },
      })

      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: inputBuffer } },
          { binding: 2, resource: { buffer: outputBuffer } },
        ],
      })

      const commandEncoder = device.createCommandEncoder()
      const pass = commandEncoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(1)
      pass.end()

      const stagingBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })
      commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 4)
      device.queue.submit([commandEncoder.finish()])

      await stagingBuffer.mapAsync(GPUMapMode.READ)
      const result = new Float32Array(stagingBuffer.getMappedRange().slice(0))
      stagingBuffer.unmap()

      // dx = 3 is within half box (5), should stay 3
      expect(result[0]).toBeCloseTo(3.0, 6)

      // Cleanup
      paramsBuffer.destroy()
      inputBuffer.destroy()
      outputBuffer.destroy()
      stagingBuffer.destroy()
    })
  })

  describe('LJ Force with PBC', () => {
    it('should compute correct force for atoms across periodic boundary', async () => {
      if (!device) return

      // Full LJ force shader with PBC
      const shaderCode = `
        struct Params {
          numAtoms: u32,
          epsilon: f32,
          sigma: f32,
          cutoffSq: f32,
          boxLx: f32,
          boxLy: f32,
          boxLz: f32,
          periodicX: u32,
          periodicY: u32,
          periodicZ: u32,
          _pad1: u32,
          _pad2: u32,
        }

        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read> positions: array<f32>;
        @group(0) @binding(2) var<storage, read_write> forces: array<f32>;
        @group(0) @binding(3) var<storage, read_write> debug: array<f32>;

        fn minimumImage(dx: f32, dy: f32, dz: f32) -> vec3<f32> {
          var r = vec3<f32>(dx, dy, dz);
          
          if (params.periodicX != 0u) {
            if (r.x > params.boxLx * 0.5) {
              r.x -= params.boxLx;
            } else if (r.x < -params.boxLx * 0.5) {
              r.x += params.boxLx;
            }
          }
          
          if (params.periodicY != 0u) {
            if (r.y > params.boxLy * 0.5) {
              r.y -= params.boxLy;
            } else if (r.y < -params.boxLy * 0.5) {
              r.y += params.boxLy;
            }
          }
          
          if (params.periodicZ != 0u) {
            if (r.z > params.boxLz * 0.5) {
              r.z -= params.boxLz;
            } else if (r.z < -params.boxLz * 0.5) {
              r.z += params.boxLz;
            }
          }
          
          return r;
        }

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let i = id.x;
          if (i >= params.numAtoms) { return; }

          let xi = positions[i * 3u];
          let yi = positions[i * 3u + 1u];
          let zi = positions[i * 3u + 2u];

          var fx: f32 = 0.0;
          var fy: f32 = 0.0;
          var fz: f32 = 0.0;

          for (var j: u32 = 0u; j < params.numAtoms; j++) {
            if (j == i) { continue; }

            let xj = positions[j * 3u];
            let yj = positions[j * 3u + 1u];
            let zj = positions[j * 3u + 2u];

            // Use minimum image
            let delta = minimumImage(xi - xj, yi - yj, zi - zj);
            let rsq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;

            // Store debug info for atom 0
            if (i == 0u) {
              debug[0] = delta.x;
              debug[1] = delta.y;
              debug[2] = delta.z;
              debug[3] = rsq;
            }

            if (rsq < params.cutoffSq && rsq > 0.01) {
              let r2inv = 1.0 / rsq;
              let r6inv = r2inv * r2inv * r2inv;
              let sigma6 = params.sigma * params.sigma * params.sigma * 
                          params.sigma * params.sigma * params.sigma;
              
              // LAMMPS formula: forcelj = r6inv * (lj1 * r6inv - lj2)
              // lj1 = 48 * epsilon * sigma^12
              // lj2 = 24 * epsilon * sigma^6
              let lj1 = 48.0 * params.epsilon * sigma6 * sigma6;
              let lj2 = 24.0 * params.epsilon * sigma6;
              let forcelj = r6inv * (lj1 * r6inv - lj2);
              let fpair = forcelj * r2inv;

              fx += delta.x * fpair;
              fy += delta.y * fpair;
              fz += delta.z * fpair;

              if (i == 0u) {
                debug[4] = fpair;
              }
            }
          }

          forces[i * 3u] = fx;
          forces[i * 3u + 1u] = fy;
          forces[i * 3u + 2u] = fz;
        }
      `

      const shaderModule = device.createShaderModule({ code: shaderCode })

      // Box 10x10x10, atoms at x=1 and x=9 (distance across PBC = 2)
      // At r=2 with sigma=1, this is repulsive (r < 2^(1/6) * sigma ≈ 1.122)
      // Wait, r=2 > 1.122, so it should be attractive!
      // Let's use sigma=1.5 so r=2 < 2^(1/6)*1.5 ≈ 1.68 -> still attractive
      // Use sigma=2.0 so r=2 < 2^(1/6)*2 ≈ 2.24 -> attractive
      // For repulsive, need r < 2^(1/6)*sigma
      
      const paramsData = new ArrayBuffer(48)
      const u32 = new Uint32Array(paramsData)
      const f32 = new Float32Array(paramsData)
      u32[0] = 2      // numAtoms
      f32[1] = 1.0    // epsilon
      f32[2] = 1.0    // sigma
      f32[3] = 25.0   // cutoffSq = 5^2
      f32[4] = 10.0   // boxLx
      f32[5] = 10.0   // boxLy
      f32[6] = 10.0   // boxLz
      u32[7] = 1      // periodicX
      u32[8] = 1      // periodicY
      u32[9] = 1      // periodicZ

      const paramsBuffer = device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(paramsBuffer, 0, paramsData)

      // Atom 0 at (1, 0, 0), Atom 1 at (9, 0, 0)
      // Across PBC, effective distance = 2
      const positions = new Float32Array([
        1, 0, 0,  // Atom 0
        9, 0, 0,  // Atom 1
      ])
      const posBuffer = device.createBuffer({
        size: 24,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(posBuffer, 0, positions)

      const forcesBuffer = device.createBuffer({
        size: 24,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      })

      const debugBuffer = device.createBuffer({
        size: 24,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      })

      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      })

      const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: 'main' },
      })

      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: posBuffer } },
          { binding: 2, resource: { buffer: forcesBuffer } },
          { binding: 3, resource: { buffer: debugBuffer } },
        ],
      })

      const commandEncoder = device.createCommandEncoder()
      const pass = commandEncoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(2)
      pass.end()

      const forcesStaging = device.createBuffer({
        size: 24,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })
      const debugStaging = device.createBuffer({
        size: 24,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })

      commandEncoder.copyBufferToBuffer(forcesBuffer, 0, forcesStaging, 0, 24)
      commandEncoder.copyBufferToBuffer(debugBuffer, 0, debugStaging, 0, 24)
      device.queue.submit([commandEncoder.finish()])

      await forcesStaging.mapAsync(GPUMapMode.READ)
      await debugStaging.mapAsync(GPUMapMode.READ)

      const forces = new Float32Array(forcesStaging.getMappedRange().slice(0))
      const debug = new Float32Array(debugStaging.getMappedRange().slice(0))

      forcesStaging.unmap()
      debugStaging.unmap()

      // Debug output
      console.log('Delta:', debug[0], debug[1], debug[2])
      console.log('rsq:', debug[3])
      console.log('fpair:', debug[4])
      console.log('Forces atom 0:', forces[0], forces[1], forces[2])
      console.log('Forces atom 1:', forces[3], forces[4], forces[5])

      // Minimum image: (1,0,0) - (9,0,0) = (-8,0,0) -> wrap to (2,0,0)
      expect(debug[0]).toBeCloseTo(2, 4)  // dx after MIC
      expect(debug[3]).toBeCloseTo(4, 4)  // rsq = 2^2 = 4

      // At r=2 with sigma=1, r > r_min (2^(1/6)*sigma ≈ 1.122)
      // So force is ATTRACTIVE (negative fpair means atoms pulled together)
      // Force on atom 0: delta * fpair = (2,0,0) * fpair
      // If attractive, fpair < 0, so fx < 0 (pulled toward atom 1)
      expect(forces[0]).toBeLessThan(0)  // Atom 0 attracted toward atom 1 (in -x dir relative to delta)
      
      // Newton's 3rd law
      expect(forces[0]).toBeCloseTo(-forces[3], 4)

      // Cleanup
      paramsBuffer.destroy()
      posBuffer.destroy()
      forcesBuffer.destroy()
      debugBuffer.destroy()
      forcesStaging.destroy()
      debugStaging.destroy()
    })

    it('should compute repulsive force when atoms are too close', async () => {
      if (!device) return

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

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let i = id.x;
          if (i >= params.numAtoms) { return; }

          let xi = positions[i * 3u];
          let yi = positions[i * 3u + 1u];
          let zi = positions[i * 3u + 2u];

          var fx: f32 = 0.0;
          var fy: f32 = 0.0;
          var fz: f32 = 0.0;

          for (var j: u32 = 0u; j < params.numAtoms; j++) {
            if (j == i) { continue; }

            let xj = positions[j * 3u];
            let yj = positions[j * 3u + 1u];
            let zj = positions[j * 3u + 2u];

            let dx = xi - xj;
            let dy = yi - yj;
            let dz = zi - zj;
            let rsq = dx * dx + dy * dy + dz * dz;

            if (rsq < params.cutoffSq && rsq > 0.001) {
              let r2inv = 1.0 / rsq;
              let r6inv = r2inv * r2inv * r2inv;
              let sigma6 = params.sigma * params.sigma * params.sigma * 
                          params.sigma * params.sigma * params.sigma;
              
              let lj1 = 48.0 * params.epsilon * sigma6 * sigma6;
              let lj2 = 24.0 * params.epsilon * sigma6;
              let forcelj = r6inv * (lj1 * r6inv - lj2);
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

      // Two atoms at r = 0.9 * sigma (inside repulsive region)
      const paramsData = new ArrayBuffer(16)
      new Uint32Array(paramsData, 0, 1)[0] = 2  // numAtoms
      new Float32Array(paramsData, 4, 3).set([1.0, 1.0, 6.25])  // epsilon, sigma, cutoffSq

      const paramsBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(paramsBuffer, 0, paramsData)

      // Atoms at r = 0.9 (repulsive)
      const positions = new Float32Array([
        0, 0, 0,    // Atom 0
        0.9, 0, 0,  // Atom 1
      ])
      const posBuffer = device.createBuffer({
        size: 24,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(posBuffer, 0, positions)

      const forcesBuffer = device.createBuffer({
        size: 24,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      })

      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      })

      const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: 'main' },
      })

      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: posBuffer } },
          { binding: 2, resource: { buffer: forcesBuffer } },
        ],
      })

      const commandEncoder = device.createCommandEncoder()
      const pass = commandEncoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(2)
      pass.end()

      const stagingBuffer = device.createBuffer({
        size: 24,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })
      commandEncoder.copyBufferToBuffer(forcesBuffer, 0, stagingBuffer, 0, 24)
      device.queue.submit([commandEncoder.finish()])

      await stagingBuffer.mapAsync(GPUMapMode.READ)
      const forces = new Float32Array(stagingBuffer.getMappedRange().slice(0))
      stagingBuffer.unmap()

      console.log('Repulsive forces:', forces[0], forces[1], forces[2], '|', forces[3], forces[4], forces[5])

      // At r = 0.9 < sigma, atoms should repel strongly
      // Atom 0 at origin, Atom 1 at +x
      // Atom 0 should be pushed in -x direction (negative force)
      // Atom 1 should be pushed in +x direction (positive force)
      expect(forces[0]).toBeLessThan(0)      // Atom 0 pushed in -x
      expect(forces[3]).toBeGreaterThan(0)   // Atom 1 pushed in +x
      expect(Math.abs(forces[0])).toBeGreaterThan(10)  // Strong repulsion

      // Cleanup
      paramsBuffer.destroy()
      posBuffer.destroy()
      forcesBuffer.destroy()
      stagingBuffer.destroy()
    })
  })

  describe('FCC Lattice Distance', () => {
    it('should verify FCC nearest neighbor distance', () => {
      // For FCC with lattice constant a:
      // Nearest neighbor distance = a / sqrt(2) ≈ 0.707 * a
      // For density 0.8 in LJ units with 4 atoms per unit cell:
      // a^3 * 0.8 = 4 -> a = (4/0.8)^(1/3) ≈ 1.71
      // NN distance = 1.71 / sqrt(2) ≈ 1.21
      
      const density = 0.8
      const atomsPerCell = 4
      const a = Math.pow(atomsPerCell / density, 1/3)
      const nnDistance = a / Math.sqrt(2)
      
      console.log('FCC lattice constant a:', a)
      console.log('FCC nearest neighbor distance:', nnDistance)
      
      // For LJ with sigma=1, the equilibrium distance is 2^(1/6) ≈ 1.122
      // NN distance should be > equilibrium to avoid huge repulsive forces
      const equilibrium = Math.pow(2, 1/6)
      console.log('LJ equilibrium distance:', equilibrium)
      
      expect(nnDistance).toBeGreaterThan(equilibrium)
    })
  })
})

