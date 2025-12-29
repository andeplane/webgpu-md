// Velocity Verlet Integrator Shader
// Two-phase integration:
// Phase 1 (initial): v += 0.5 * dt * f / m, then x += dt * v, apply PBC
// Phase 2 (final): v += 0.5 * dt * f / m

struct IntegratorParams {
  numAtoms: u32,
  halfDt: f32,       // 0.5 * dt
  dt: f32,           // timestep
  _padding1: u32,
  // Box parameters
  originX: f32,
  originY: f32,
  originZ: f32,
  boxLx: f32,
  boxLy: f32,
  boxLz: f32,
  periodicX: u32,
  periodicY: u32,
  periodicZ: u32,
  _padding2: u32,
  _padding3: u32,
  _padding4: u32,
}

@group(0) @binding(0) var<uniform> params: IntegratorParams;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(3) var<storage, read> forces: array<f32>;
@group(0) @binding(4) var<storage, read> masses: array<f32>;

// Apply periodic boundary conditions
fn applyPBC(x: f32, y: f32, z: f32) -> vec3<f32> {
  var result = vec3<f32>(x, y, z);
  
  // Wrap into box
  if (params.periodicX != 0u) {
    let px = result.x - params.originX;
    result.x = params.originX + px - floor(px / params.boxLx) * params.boxLx;
  }
  
  if (params.periodicY != 0u) {
    let py = result.y - params.originY;
    result.y = params.originY + py - floor(py / params.boxLy) * params.boxLy;
  }
  
  if (params.periodicZ != 0u) {
    let pz = result.z - params.originZ;
    result.z = params.originZ + pz - floor(pz / params.boxLz) * params.boxLz;
  }
  
  return result;
}

// Phase 1: Update velocities (first half) and positions
// v += 0.5 * dt * f / m
// x += dt * v
// Apply PBC to positions
@compute @workgroup_size(256)
fn integrateInitial(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let i = global_id.x;
  
  if (i >= params.numAtoms) {
    return;
  }
  
  // Load current state
  let vx = velocities[i * 3u + 0u];
  let vy = velocities[i * 3u + 1u];
  let vz = velocities[i * 3u + 2u];
  
  let fx = forces[i * 3u + 0u];
  let fy = forces[i * 3u + 1u];
  let fz = forces[i * 3u + 2u];
  
  let mass = masses[i];
  let invMass = 1.0 / mass;
  
  // Update velocities: v += 0.5 * dt * f / m
  let newVx = vx + params.halfDt * fx * invMass;
  let newVy = vy + params.halfDt * fy * invMass;
  let newVz = vz + params.halfDt * fz * invMass;
  
  // Store updated velocities
  velocities[i * 3u + 0u] = newVx;
  velocities[i * 3u + 1u] = newVy;
  velocities[i * 3u + 2u] = newVz;
  
  // Load current positions
  let x = positions[i * 3u + 0u];
  let y = positions[i * 3u + 1u];
  let z = positions[i * 3u + 2u];
  
  // Update positions: x += dt * v
  var newX = x + params.dt * newVx;
  var newY = y + params.dt * newVy;
  var newZ = z + params.dt * newVz;
  
  // Apply periodic boundary conditions
  let wrapped = applyPBC(newX, newY, newZ);
  
  // Store updated positions
  positions[i * 3u + 0u] = wrapped.x;
  positions[i * 3u + 1u] = wrapped.y;
  positions[i * 3u + 2u] = wrapped.z;
}

// Phase 2: Update velocities (second half)
// v += 0.5 * dt * f / m
// Called after force computation
@compute @workgroup_size(256)
fn integrateFinal(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let i = global_id.x;
  
  if (i >= params.numAtoms) {
    return;
  }
  
  // Load current velocities and new forces
  let vx = velocities[i * 3u + 0u];
  let vy = velocities[i * 3u + 1u];
  let vz = velocities[i * 3u + 2u];
  
  let fx = forces[i * 3u + 0u];
  let fy = forces[i * 3u + 1u];
  let fz = forces[i * 3u + 2u];
  
  let mass = masses[i];
  let invMass = 1.0 / mass;
  
  // Update velocities: v += 0.5 * dt * f / m
  velocities[i * 3u + 0u] = vx + params.halfDt * fx * invMass;
  velocities[i * 3u + 1u] = vy + params.halfDt * fy * invMass;
  velocities[i * 3u + 2u] = vz + params.halfDt * fz * invMass;
}

