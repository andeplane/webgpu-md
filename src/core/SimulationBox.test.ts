import { describe, it, expect } from 'vitest'
import * as THREE from 'three'
import { SimulationBox } from './SimulationBox'

describe('SimulationBox', () => {
  describe('construction', () => {
    it('should create default orthogonal box', () => {
      // Arrange & Act
      const box = new SimulationBox()

      // Assert
      expect(box.dimensions).toEqual([10, 10, 10])
      expect(box.tilts).toEqual([0, 0, 0])
      expect(box.isTriclinic).toBe(false)
      expect(box.periodic).toEqual([true, true, true])
    })

    it.each([
      { lx: 5, ly: 10, lz: 15 },
      { lx: 1, ly: 1, lz: 1 },
      { lx: 100, ly: 100, lz: 100 },
    ])('should create orthogonal box with dimensions $lx x $ly x $lz', ({ lx, ly, lz }) => {
      // Arrange & Act
      const box = SimulationBox.fromDimensions(lx, ly, lz)

      // Assert
      expect(box.dimensions).toEqual([lx, ly, lz])
      expect(box.isTriclinic).toBe(false)
    })

    it('should create triclinic box with tilts using fromBounds', () => {
      // Arrange
      const xlo = 0, xhi = 10, ylo = 0, yhi = 10, zlo = 0, zhi = 10
      const xy = 1, xz = 0.5, yz = 0.5

      // Act
      const box = SimulationBox.fromBounds(xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz)

      // Assert
      expect(box.dimensions).toEqual([10, 10, 10])
      expect(box.tilts).toEqual([xy, xz, yz])
      expect(box.isTriclinic).toBe(true)
    })
  })

  describe('volume', () => {
    it.each([
      { lx: 10, ly: 10, lz: 10, expected: 1000 },
      { lx: 5, ly: 5, lz: 5, expected: 125 },
      { lx: 2, ly: 3, lz: 4, expected: 24 },
    ])('should calculate volume $expected for box $lx x $ly x $lz', ({ lx, ly, lz, expected }) => {
      // Arrange
      const box = SimulationBox.fromDimensions(lx, ly, lz)

      // Act
      const volume = box.volume

      // Assert
      expect(volume).toBeCloseTo(expected, 10)
    })
  })

  describe('center', () => {
    it('should calculate center of orthogonal box at origin', () => {
      // Arrange
      const box = SimulationBox.fromDimensions(10, 10, 10)

      // Act
      const center = box.getCenter()

      // Assert
      expect(center.x).toBeCloseTo(5, 10)
      expect(center.y).toBeCloseTo(5, 10)
      expect(center.z).toBeCloseTo(5, 10)
    })

    it('should calculate center with non-zero origin', () => {
      // Arrange
      const box = new SimulationBox(
        new THREE.Vector3(10, 0, 0),
        new THREE.Vector3(0, 10, 0),
        new THREE.Vector3(0, 0, 10),
        new THREE.Vector3(-5, -5, -5)
      )

      // Act
      const center = box.getCenter()

      // Assert
      expect(center.x).toBeCloseTo(0, 10)
      expect(center.y).toBeCloseTo(0, 10)
      expect(center.z).toBeCloseTo(0, 10)
    })
  })

  describe('clone', () => {
    it('should create independent copy', () => {
      // Arrange
      const original = SimulationBox.fromDimensions(10, 20, 30)
      original.periodic = [true, false, true]

      // Act
      const cloned = original.clone()

      // Assert
      expect(cloned.dimensions).toEqual(original.dimensions)
      expect(cloned.periodic).toEqual(original.periodic)
      
      // Verify independence
      cloned.a.x = 999
      expect(original.a.x).toBe(10)
    })
  })

  describe('toGPUData', () => {
    it('should pack box data for GPU uniform buffer', () => {
      // Arrange
      const box = SimulationBox.fromDimensions(10, 20, 30)
      box.origin = new THREE.Vector3(1, 2, 3)

      // Act
      const data = box.toGPUData()

      // Assert
      expect(data).toBeInstanceOf(Float32Array)
      expect(data.length).toBe(24) // 6 vec4s

      // GPU layout: origin, a, b, c, dimensions, periodic
      // First vec4: origin + padding
      expect(data[0]).toBe(1)  // origin.x
      expect(data[1]).toBe(2)  // origin.y
      expect(data[2]).toBe(3)  // origin.z
      expect(data[3]).toBe(0)  // padding

      // Second vec4: a vector + padding
      expect(data[4]).toBe(10) // a.x
      expect(data[5]).toBe(0)  // a.y
      expect(data[6]).toBe(0)  // a.z

      // Third vec4: b vector + padding
      expect(data[8]).toBe(0)   // b.x
      expect(data[9]).toBe(20)  // b.y
      expect(data[10]).toBe(0)  // b.z

      // Fourth vec4: c vector + padding
      expect(data[12]).toBe(0)  // c.x
      expect(data[13]).toBe(0)  // c.y
      expect(data[14]).toBe(30) // c.z

      // Fifth vec4: dimensions + padding
      expect(data[16]).toBe(10) // lx
      expect(data[17]).toBe(20) // ly
      expect(data[18]).toBe(30) // lz

      // Sixth vec4: periodic flags + padding
      expect(data[20]).toBe(1)  // periodicX (true -> 1)
      expect(data[21]).toBe(1)  // periodicY (true -> 1)
      expect(data[22]).toBe(1)  // periodicZ (true -> 1)
    })
  })
})

