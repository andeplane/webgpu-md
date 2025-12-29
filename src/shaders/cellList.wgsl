// Cell List Construction Shader
// Bins atoms into spatial cells for efficient neighbor searching

struct SimParams {
  numAtoms: u32,
  numCellsX: u32,
  numCellsY: u32,
  numCellsZ: u32,
  cellSizeX: f32,
  cellSizeY: f32,
  cellSizeZ: f32,
  maxAtomsPerCell: u32,
  // Box parameters
  originX: f32,
  originY: f32,
  originZ: f32,
  boxLx: f32,
  boxLy: f32,
  boxLz: f32,
  _padding1: f32,
  _padding2: f32,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cellAtoms: array<u32>;
@group(0) @binding(4) var<storage, read_write> atomCell: array<u32>;

// Convert 3D cell indices to linear index
fn cellIndex(ix: u32, iy: u32, iz: u32) -> u32 {
  return ix + iy * params.numCellsX + iz * params.numCellsX * params.numCellsY;
}

// Get cell indices for a position
fn positionToCell(x: f32, y: f32, z: f32) -> vec3<u32> {
  // Apply minimum image - wrap position into box
  var px = x - params.originX;
  var py = y - params.originY;
  var pz = z - params.originZ;
  
  // Wrap into box (assuming periodic)
  px = px - floor(px / params.boxLx) * params.boxLx;
  py = py - floor(py / params.boxLy) * params.boxLy;
  pz = pz - floor(pz / params.boxLz) * params.boxLz;
  
  // Compute cell indices
  let ix = min(u32(px / params.cellSizeX), params.numCellsX - 1u);
  let iy = min(u32(py / params.cellSizeY), params.numCellsY - 1u);
  let iz = min(u32(pz / params.cellSizeZ), params.numCellsZ - 1u);
  
  return vec3<u32>(ix, iy, iz);
}

@compute @workgroup_size(256)
fn resetCells(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let numCells = params.numCellsX * params.numCellsY * params.numCellsZ;
  
  if (idx < numCells) {
    atomicStore(&cellCounts[idx], 0u);
  }
}

@compute @workgroup_size(256)
fn binAtoms(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let atomIdx = global_id.x;
  
  if (atomIdx >= params.numAtoms) {
    return;
  }
  
  // Get atom position
  let x = positions[atomIdx * 3u + 0u];
  let y = positions[atomIdx * 3u + 1u];
  let z = positions[atomIdx * 3u + 2u];
  
  // Get cell for this atom
  let cell = positionToCell(x, y, z);
  let cellIdx = cellIndex(cell.x, cell.y, cell.z);
  
  // Store which cell this atom belongs to
  atomCell[atomIdx] = cellIdx;
  
  // Atomically increment cell count and get slot
  let slot = atomicAdd(&cellCounts[cellIdx], 1u);
  
  // Store atom in cell's atom list (if there's room)
  if (slot < params.maxAtomsPerCell) {
    let offset = cellIdx * params.maxAtomsPerCell + slot;
    cellAtoms[offset] = atomIdx;
  }
  // Note: If slot >= maxAtomsPerCell, atom is not stored (overflow)
  // In production, should track overflow and resize
}

