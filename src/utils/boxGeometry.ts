import * as THREE from 'three'
import type { SimulationBox } from '../core/SimulationBox'

// Constants for box geometry rendering
const RADIUS_SCALE_FACTOR = 0.0015
const MIN_RADIUS = 0.05
const CYLINDER_RADIAL_SEGMENTS = 8
const ZERO_LENGTH_THRESHOLD = 1e-4
const MIN_NORMALIZED_LENGTH = 0.001

/**
 * Calculate an appropriate radius for box wireframe edges
 */
export function calculateBoxRadius(box: SimulationBox): number {
  const [lx, ly, lz] = box.dimensions
  const avgLength = (lx + ly + lz) / 3
  return Math.max(avgLength * RADIUS_SCALE_FACTOR, MIN_RADIUS)
}

/**
 * Creates a THREE.Group of cylinders for a parallelepiped wireframe
 * Handles both orthogonal and triclinic (non-orthogonal) simulation boxes
 */
export function createBoxGeometry(box: SimulationBox, radius?: number): THREE.Group {
  const r = radius ?? calculateBoxRadius(box)
  
  // Get basis vectors
  const a = box.a.clone()
  const b = box.b.clone()
  const c = box.c.clone()
  const origin = box.origin.clone()

  // Compute the 8 vertices of the parallelepiped
  const vertices: THREE.Vector3[] = [
    origin.clone(),                           // v0
    origin.clone().add(a),                    // v1
    origin.clone().add(b),                    // v2
    origin.clone().add(c),                    // v3
    origin.clone().add(a).add(b),             // v4
    origin.clone().add(a).add(c),             // v5
    origin.clone().add(b).add(c),             // v6
    origin.clone().add(a).add(b).add(c),      // v7
  ]

  // Define the 12 edges of the parallelepiped
  const edges: [number, number][] = [
    // Bottom face
    [0, 1], [1, 4], [4, 2], [2, 0],
    // Top face
    [3, 5], [5, 7], [7, 6], [6, 3],
    // Vertical edges
    [0, 3], [1, 5], [4, 7], [2, 6],
  ]

  const group = new THREE.Group()
  group.name = 'simulation-box'

  // Create material - white wireframe
  const material = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.6,
  })

  // Create a cylinder for each edge
  for (const [i, j] of edges) {
    const start = vertices[i]
    const end = vertices[j]
    const direction = new THREE.Vector3().subVectors(end, start)
    const length = direction.length()

    if (length < ZERO_LENGTH_THRESHOLD) continue

    // Create cylinder geometry (default orientation is along Y-axis)
    const geometry = new THREE.CylinderGeometry(r, r, length, CYLINDER_RADIAL_SEGMENTS)
    const cylinder = new THREE.Mesh(geometry, material)

    // Position at midpoint
    const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5)
    cylinder.position.copy(midpoint)

    // Orient cylinder along the edge direction
    const targetAxis = direction.clone().normalize()

    if (
      !isFinite(targetAxis.x) ||
      !isFinite(targetAxis.y) ||
      !isFinite(targetAxis.z) ||
      targetAxis.length() < MIN_NORMALIZED_LENGTH
    ) {
      geometry.dispose()
      continue
    }

    // Align cylinder to edge direction
    cylinder.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), targetAxis)

    if (
      !isFinite(cylinder.quaternion.x) ||
      !isFinite(cylinder.quaternion.y) ||
      !isFinite(cylinder.quaternion.z) ||
      !isFinite(cylinder.quaternion.w)
    ) {
      cylinder.quaternion.set(0, 0, 0, 1)
    }

    group.add(cylinder)
  }

  return group
}

/**
 * Dispose of a box geometry group and its resources
 */
export function disposeBoxGeometry(group: THREE.Group): void {
  group.traverse((obj) => {
    if (obj instanceof THREE.Mesh) {
      obj.geometry.dispose()
      if (obj.material instanceof THREE.Material) {
        obj.material.dispose()
      }
    }
  })
  group.clear()
}

