// Neighbor List Builder Shader
// Builds Verlet neighbor lists using cell lists for efficient searching
// Uses FULL neighbor list (no Newton's 3rd law) for simpler parallel computation

struct NeighParams {
  numAtoms: u32,
  numCellsX: u32,
  numCellsY: u32,
  numCellsZ: u32,
  maxNeighbors: u32,
  cutoffSq: f32,
  maxAtomsPerCell: u32,
  _padding: u32,
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

@group(0) @binding(0) var<uniform> params: NeighParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> cellCounts: array<u32>;
@group(0) @binding(3) var<storage, read> cellAtoms: array<u32>;
@group(0) @binding(4) var<storage, read> atomCell: array<u32>;
@group(0) @binding(5) var<storage, read_write> neighborList: array<u32>;
@group(0) @binding(6) var<storage, read_write> numNeighbors: array<u32>;

// Convert 3D cell indices to linear index
fn cellIndex(ix: i32, iy: i32, iz: i32) -> u32 {
  // Handle periodic wrapping
  var cx = ix;
  var cy = iy;
  var cz = iz;
  
  if (params.periodicX != 0u) {
    cx = (cx + i32(params.numCellsX)) % i32(params.numCellsX);
  }
  if (params.periodicY != 0u) {
    cy = (cy + i32(params.numCellsY)) % i32(params.numCellsY);
  }
  if (params.periodicZ != 0u) {
    cz = (cz + i32(params.numCellsZ)) % i32(params.numCellsZ);
  }
  
  return u32(cx) + u32(cy) * params.numCellsX + u32(cz) * params.numCellsX * params.numCellsY;
}

// Check if cell index is valid (for non-periodic boundaries)
fn isCellValid(ix: i32, iy: i32, iz: i32) -> bool {
  if (params.periodicX == 0u && (ix < 0 || ix >= i32(params.numCellsX))) {
    return false;
  }
  if (params.periodicY == 0u && (iy < 0 || iy >= i32(params.numCellsY))) {
    return false;
  }
  if (params.periodicZ == 0u && (iz < 0 || iz >= i32(params.numCellsZ))) {
    return false;
  }
  return true;
}

// Minimum image convention for periodic boundaries
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
fn buildNeighborList(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let i = global_id.x;
  
  if (i >= params.numAtoms) {
    return;
  }
  
  // Get atom i's position
  let xi = positions[i * 3u + 0u];
  let yi = positions[i * 3u + 1u];
  let zi = positions[i * 3u + 2u];
  
  // Get atom i's cell
  let iCell = atomCell[i];
  let icx = i32(iCell % params.numCellsX);
  let icy = i32((iCell / params.numCellsX) % params.numCellsY);
  let icz = i32(iCell / (params.numCellsX * params.numCellsY));
  
  var nNeigh: u32 = 0u;
  let baseOffset = i * params.maxNeighbors;
  
  // Loop over 27 neighboring cells (3x3x3 stencil)
  for (var dz: i32 = -1; dz <= 1; dz++) {
    for (var dy: i32 = -1; dy <= 1; dy++) {
      for (var dx: i32 = -1; dx <= 1; dx++) {
        let jcx = icx + dx;
        let jcy = icy + dy;
        let jcz = icz + dz;
        
        // Skip invalid cells (non-periodic boundaries)
        if (!isCellValid(jcx, jcy, jcz)) {
          continue;
        }
        
        let jCell = cellIndex(jcx, jcy, jcz);
        let cellCount = cellCounts[jCell];
        let cellOffset = jCell * params.maxAtomsPerCell;
        
        // Loop over atoms in neighbor cell
        for (var k: u32 = 0u; k < cellCount && k < params.maxAtomsPerCell; k++) {
          let j = cellAtoms[cellOffset + k];
          
          // Skip self-interaction
          if (j == i) {
            continue;
          }
          
          // Get atom j's position
          let xj = positions[j * 3u + 0u];
          let yj = positions[j * 3u + 1u];
          let zj = positions[j * 3u + 2u];
          
          // Calculate distance with minimum image
          let delta = minimumImage(xi - xj, yi - yj, zi - zj);
          let rsq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
          
          // Check if within cutoff
          if (rsq < params.cutoffSq) {
            if (nNeigh < params.maxNeighbors) {
              neighborList[baseOffset + nNeigh] = j;
              nNeigh++;
            }
            // Note: if nNeigh >= maxNeighbors, neighbor is dropped (overflow)
          }
        }
      }
    }
  }
  
  // Store neighbor count for this atom
  numNeighbors[i] = nNeigh;
}

