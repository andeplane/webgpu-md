// Lennard-Jones Pair Style Compute Shader
// Computes LJ forces using neighbor lists
// Following LAMMPS pair_lj_cut formula

struct LJParams {
  numAtoms: u32,
  numTypes: u32,
  maxNeighbors: u32,
  computeEnergy: u32,
  // Box parameters for minimum image
  boxLx: f32,
  boxLy: f32,
  boxLz: f32,
  periodicX: u32,
  periodicY: u32,
  periodicZ: u32,
  _padding1: u32,
  _padding2: u32,
}

// LJ coefficients per type pair:
// [0] = lj1 = 48 * epsilon * sigma^12
// [1] = lj2 = 24 * epsilon * sigma^6
// [2] = lj3 = 4 * epsilon * sigma^12 (for energy)
// [3] = lj4 = 4 * epsilon * sigma^6 (for energy)
// [4] = cutsq = cutoff^2
// [5] = offset (energy offset at cutoff)
// Stored as: coeffs[typeI * numTypes + typeJ]
struct LJCoeffs {
  lj1: f32,
  lj2: f32,
  lj3: f32,
  lj4: f32,
  cutsq: f32,
  offset: f32,
  _padding1: f32,
  _padding2: f32,
}

@group(0) @binding(0) var<uniform> params: LJParams;
@group(0) @binding(1) var<storage, read> coeffs: array<LJCoeffs>;
@group(0) @binding(2) var<storage, read> positions: array<f32>;
@group(0) @binding(3) var<storage, read> types: array<u32>;
@group(0) @binding(4) var<storage, read> neighborList: array<u32>;
@group(0) @binding(5) var<storage, read> numNeighbors: array<u32>;
@group(0) @binding(6) var<storage, read_write> forces: array<f32>;
@group(0) @binding(7) var<storage, read_write> energy: array<f32>;

// Minimum image convention
fn minimumImage(dx: f32, dy: f32, dz: f32) -> vec3<f32> {
  var result = vec3<f32>(dx, dy, dz);
  
  if (params.periodicX != 0u) {
    if (result.x > params.boxLx * 0.5) {
      result.x -= params.boxLx;
    } else if (result.x < -params.boxLx * 0.5) {
      result.x += params.boxLx;
    }
  }
  
  if (params.periodicY != 0u) {
    if (result.y > params.boxLy * 0.5) {
      result.y -= params.boxLy;
    } else if (result.y < -params.boxLy * 0.5) {
      result.y += params.boxLy;
    }
  }
  
  if (params.periodicZ != 0u) {
    if (result.z > params.boxLz * 0.5) {
      result.z -= params.boxLz;
    } else if (result.z < -params.boxLz * 0.5) {
      result.z += params.boxLz;
    }
  }
  
  return result;
}

@compute @workgroup_size(64)
fn zeroForces(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let i = global_id.x;
  
  if (i >= params.numAtoms) {
    return;
  }
  
  forces[i * 3u + 0u] = 0.0;
  forces[i * 3u + 1u] = 0.0;
  forces[i * 3u + 2u] = 0.0;
  
  if (params.computeEnergy != 0u) {
    energy[i] = 0.0;
  }
}

@compute @workgroup_size(64)
fn computeForces(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let i = global_id.x;
  
  if (i >= params.numAtoms) {
    return;
  }
  
  // Get atom i's position and type
  let xi = positions[i * 3u + 0u];
  let yi = positions[i * 3u + 1u];
  let zi = positions[i * 3u + 2u];
  let itype = types[i];
  
  // Accumulate forces and energy
  var fx: f32 = 0.0;
  var fy: f32 = 0.0;
  var fz: f32 = 0.0;
  var evdwl: f32 = 0.0;
  
  // Get neighbor count and base offset
  let nNeigh = numNeighbors[i];
  let baseOffset = i * params.maxNeighbors;
  
  // Loop over neighbors
  for (var n: u32 = 0u; n < nNeigh; n++) {
    let j = neighborList[baseOffset + n];
    let jtype = types[j];
    
    // Get atom j's position
    let xj = positions[j * 3u + 0u];
    let yj = positions[j * 3u + 1u];
    let zj = positions[j * 3u + 2u];
    
    // Calculate distance with minimum image
    let delta = minimumImage(xi - xj, yi - yj, zi - zj);
    let rsq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
    
    // Get coefficients for this type pair
    let coeffIdx = itype * params.numTypes + jtype;
    let coeff = coeffs[coeffIdx];
    
    // Check cutoff
    if (rsq < coeff.cutsq) {
      // LJ force calculation (LAMMPS formula)
      let r2inv = 1.0 / rsq;
      let r6inv = r2inv * r2inv * r2inv;
      
      // forcelj = r6inv * (lj1 * r6inv - lj2)
      let forcelj = r6inv * (coeff.lj1 * r6inv - coeff.lj2);
      
      // fpair = forcelj * r2inv
      let fpair = forcelj * r2inv;
      
      // Accumulate force on atom i
      fx += delta.x * fpair;
      fy += delta.y * fpair;
      fz += delta.z * fpair;
      
      // Energy calculation (if requested)
      if (params.computeEnergy != 0u) {
        // evdwl = r6inv * (lj3 * r6inv - lj4) - offset
        // Divide by 2 since we're using full neighbor list (each pair counted twice)
        evdwl += 0.5 * (r6inv * (coeff.lj3 * r6inv - coeff.lj4) - coeff.offset);
      }
    }
  }
  
  // Store accumulated force
  forces[i * 3u + 0u] = fx;
  forces[i * 3u + 1u] = fy;
  forces[i * 3u + 2u] = fz;
  
  // Store energy (if computed)
  if (params.computeEnergy != 0u) {
    energy[i] = evdwl;
  }
}

