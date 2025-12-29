import { describe, it, expect } from 'vitest'
import * as THREE from 'three'
import { calculateBoxRadius, createBoxGeometry, disposeBoxGeometry } from './boxGeometry'
import { SimulationBox } from '../core/SimulationBox'

describe('boxGeometry utilities', () => {
  describe('calculateBoxRadius', () => {
    it.each([
      { lx: 10, ly: 10, lz: 10, minExpected: 0.05 },
      { lx: 100, ly: 100, lz: 100, minExpected: 0.15 },
      { lx: 1, ly: 1, lz: 1, minExpected: 0.05 },
    ])('should calculate radius for $lx x $ly x $lz box', ({ lx, ly, lz, minExpected }) => {
      // Arrange
      const box = SimulationBox.fromDimensions(lx, ly, lz)

      // Act
      const radius = calculateBoxRadius(box)

      // Assert
      expect(radius).toBeGreaterThanOrEqual(minExpected)
      expect(radius).toBeLessThan(lx) // Radius should be small relative to box
    })

    it('should enforce minimum radius for tiny boxes', () => {
      // Arrange
      const box = SimulationBox.fromDimensions(0.1, 0.1, 0.1)

      // Act
      const radius = calculateBoxRadius(box)

      // Assert
      expect(radius).toBe(0.05) // MIN_RADIUS
    })
  })

  describe('createBoxGeometry', () => {
    it('should create a THREE.Group with 12 edges', () => {
      // Arrange
      const box = SimulationBox.fromDimensions(10, 10, 10)

      // Act
      const group = createBoxGeometry(box)

      // Assert
      expect(group).toBeInstanceOf(THREE.Group)
      expect(group.name).toBe('simulation-box')
      expect(group.children.length).toBe(12) // 12 edges of a cube
    })

    it('should handle non-orthogonal (triclinic) boxes', () => {
      // Arrange
      const box = SimulationBox.fromBounds(0, 10, 0, 10, 0, 10, 2, 1, 1)

      // Act
      const group = createBoxGeometry(box)

      // Assert
      expect(group.children.length).toBe(12)
    })

    it('should use custom radius when provided', () => {
      // Arrange
      const box = SimulationBox.fromDimensions(10, 10, 10)
      const customRadius = 0.5

      // Act
      const group = createBoxGeometry(box, customRadius)

      // Assert
      const firstChild = group.children[0] as THREE.Mesh
      const geometry = firstChild.geometry as THREE.CylinderGeometry
      expect(geometry.parameters.radiusTop).toBe(customRadius)
    })

    it('should position edges correctly for orthogonal box', () => {
      // Arrange
      const box = SimulationBox.fromDimensions(10, 10, 10)

      // Act
      const group = createBoxGeometry(box)

      // Assert - all meshes should be positioned within box bounds
      group.children.forEach((child) => {
        const mesh = child as THREE.Mesh
        // Positions should be within extended bounds (midpoints of edges)
        expect(mesh.position.x).toBeGreaterThanOrEqual(-1)
        expect(mesh.position.x).toBeLessThanOrEqual(11)
        expect(mesh.position.y).toBeGreaterThanOrEqual(-1)
        expect(mesh.position.y).toBeLessThanOrEqual(11)
        expect(mesh.position.z).toBeGreaterThanOrEqual(-1)
        expect(mesh.position.z).toBeLessThanOrEqual(11)
      })
    })

    it('should handle box with non-zero origin', () => {
      // Arrange
      const box = new SimulationBox(
        new THREE.Vector3(10, 0, 0),
        new THREE.Vector3(0, 10, 0),
        new THREE.Vector3(0, 0, 10),
        new THREE.Vector3(-5, -5, -5)
      )

      // Act
      const group = createBoxGeometry(box)

      // Assert
      expect(group.children.length).toBe(12)
      
      // First vertex should be at origin (-5, -5, -5)
      // Check that some edges pass through negative coordinates
      const hasNegativeCoord = group.children.some((child) => {
        const mesh = child as THREE.Mesh
        return mesh.position.x < 0 || mesh.position.y < 0 || mesh.position.z < 0
      })
      expect(hasNegativeCoord).toBe(true)
    })
  })

  describe('disposeBoxGeometry', () => {
    it('should clear all children from group', () => {
      // Arrange
      const box = SimulationBox.fromDimensions(10, 10, 10)
      const group = createBoxGeometry(box)
      expect(group.children.length).toBe(12)

      // Act
      disposeBoxGeometry(group)

      // Assert
      expect(group.children.length).toBe(0)
    })

    it('should not throw on empty group', () => {
      // Arrange
      const group = new THREE.Group()

      // Act & Assert
      expect(() => disposeBoxGeometry(group)).not.toThrow()
    })
  })
})

