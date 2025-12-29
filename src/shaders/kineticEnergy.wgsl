// Kinetic Energy Compute Shader
// Computes KE = 0.5 * sum(m_i * v_i^2) using parallel reduction

struct KEParams {
  numAtoms: u32,
  _padding1: u32,
  _padding2: u32,
  _padding3: u32,
}

@group(0) @binding(0) var<uniform> params: KEParams;
@group(0) @binding(1) var<storage, read> velocities: array<f32>;
@group(0) @binding(2) var<storage, read> masses: array<f32>;
@group(0) @binding(3) var<storage, read_write> perAtomKE: array<f32>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;

// Phase 1: Compute per-atom kinetic energy
@compute @workgroup_size(256)
fn computePerAtomKE(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let i = global_id.x;
  
  if (i >= params.numAtoms) {
    return;
  }
  
  let vx = velocities[i * 3u + 0u];
  let vy = velocities[i * 3u + 1u];
  let vz = velocities[i * 3u + 2u];
  let v2 = vx * vx + vy * vy + vz * vz;
  let mass = masses[i];
  
  perAtomKE[i] = 0.5 * mass * v2;
}

// Phase 2: Parallel reduction to sum KE
// This is a simple sequential reduction for now - could be optimized with workgroup shared memory
var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn reduceKE(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let tid = local_id.x;
  let gid = global_id.x;
  
  // Load data into shared memory
  if (gid < params.numAtoms) {
    shared_data[tid] = perAtomKE[gid];
  } else {
    shared_data[tid] = 0.0;
  }
  
  workgroupBarrier();
  
  // Parallel reduction in shared memory
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      shared_data[tid] += shared_data[tid + s];
    }
    workgroupBarrier();
  }
  
  // Write result from first thread of each workgroup
  if (tid == 0u) {
    result[workgroup_id.x] = shared_data[0];
  }
}

