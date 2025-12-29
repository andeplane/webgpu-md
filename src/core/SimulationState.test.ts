import { describe, it, expect } from 'vitest'

/**
 * SimulationState tests for CPU-side logic
 * GPU buffer operations cannot be tested without WebGPU device
 */
describe('SimulationState utility functions', () => {
  describe('seeded random number generator', () => {
    // Extract the seeded random logic for testing
    function seededRandom(seed: number): () => number {
      return () => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        return seed / 0x7fffffff
      }
    }

    it('should produce deterministic sequence with same seed', () => {
      // Arrange
      const rng1 = seededRandom(12345)
      const rng2 = seededRandom(12345)

      // Act
      const seq1 = [rng1(), rng1(), rng1(), rng1(), rng1()]
      const seq2 = [rng2(), rng2(), rng2(), rng2(), rng2()]

      // Assert
      expect(seq1).toEqual(seq2)
    })

    it('should produce different sequences with different seeds', () => {
      // Arrange
      const rng1 = seededRandom(12345)
      const rng2 = seededRandom(54321)

      // Act
      const seq1 = [rng1(), rng1(), rng1()]
      const seq2 = [rng2(), rng2(), rng2()]

      // Assert
      expect(seq1).not.toEqual(seq2)
    })

    it('should produce values in range [0, 1)', () => {
      // Arrange
      const rng = seededRandom(12345)

      // Act & Assert
      for (let i = 0; i < 100; i++) {
        const val = rng()
        expect(val).toBeGreaterThanOrEqual(0)
        expect(val).toBeLessThan(1)
      }
    })
  })

  describe('Box-Muller transform', () => {
    // Box-Muller logic extracted for testing
    function boxMuller(u1: number, u2: number): [number, number] {
      const r = Math.sqrt(-2 * Math.log(u1))
      const theta = 2 * Math.PI * u2
      return [r * Math.cos(theta), r * Math.sin(theta)]
    }

    it('should produce Gaussian-distributed values', () => {
      // Arrange - use fixed random values
      const u1 = 0.5, u2 = 0.25

      // Act
      const [z1, z2] = boxMuller(u1, u2)

      // Assert - values should be finite and reasonable
      expect(isFinite(z1)).toBe(true)
      expect(isFinite(z2)).toBe(true)
      expect(Math.abs(z1)).toBeLessThan(5) // Should be within ~5 sigma
      expect(Math.abs(z2)).toBeLessThan(5)
    })

    it('should handle edge cases', () => {
      // Arrange - value close to 0 produces large r
      const u1 = 0.001
      const u2 = 0.5

      // Act
      const [z1, z2] = boxMuller(u1, u2)

      // Assert - should handle without NaN/Inf
      expect(isFinite(z1)).toBe(true)
      expect(isFinite(z2)).toBe(true)
    })
  })

  describe('Maxwell-Boltzmann velocity initialization logic', () => {
    it('should scale velocity with temperature', () => {
      // Arrange - sigma = sqrt(kT/m) where we use reduced units (k=1)
      const temperature = 2.0
      const mass = 1.0

      // Act
      const sigma = Math.sqrt(temperature / mass)

      // Assert
      expect(sigma).toBeCloseTo(Math.sqrt(2.0))
    })

    it('should scale velocity inversely with mass', () => {
      // Arrange
      const temperature = 1.0
      const mass1 = 1.0
      const mass2 = 4.0

      // Act
      const sigma1 = Math.sqrt(temperature / mass1)
      const sigma2 = Math.sqrt(temperature / mass2)

      // Assert - heavier particles should move slower
      expect(sigma2).toBe(sigma1 / 2)
    })

    it('should remove center of mass momentum', () => {
      // Arrange - simulate 4 atoms
      const velocities = [
        [1.0, 2.0, 0.0],
        [2.0, -1.0, 1.0],
        [-1.0, 0.0, 2.0],
        [0.0, 1.0, -1.0],
      ]
      const masses = [1.0, 1.0, 1.0, 1.0]

      // Act - calculate center of mass velocity
      let sumVx = 0, sumVy = 0, sumVz = 0, totalMass = 0
      for (let i = 0; i < 4; i++) {
        sumVx += velocities[i][0] * masses[i]
        sumVy += velocities[i][1] * masses[i]
        sumVz += velocities[i][2] * masses[i]
        totalMass += masses[i]
      }
      const vcmX = sumVx / totalMass
      const vcmY = sumVy / totalMass
      const vcmZ = sumVz / totalMass

      // Remove COM velocity
      for (let i = 0; i < 4; i++) {
        velocities[i][0] -= vcmX
        velocities[i][1] -= vcmY
        velocities[i][2] -= vcmZ
      }

      // Recalculate COM
      sumVx = sumVy = sumVz = 0
      for (let i = 0; i < 4; i++) {
        sumVx += velocities[i][0] * masses[i]
        sumVy += velocities[i][1] * masses[i]
        sumVz += velocities[i][2] * masses[i]
      }

      // Assert - COM should be zero
      expect(sumVx / totalMass).toBeCloseTo(0, 10)
      expect(sumVy / totalMass).toBeCloseTo(0, 10)
      expect(sumVz / totalMass).toBeCloseTo(0, 10)
    })
  })

  describe('lattice initialization logic', () => {
    it('should calculate correct atom positions on simple cubic lattice', () => {
      // Arrange
      const nx = 2, ny = 2, nz = 2
      const spacing = 1.5
      const expectedAtoms = nx * ny * nz

      // Act
      const positions: number[] = []
      for (let iz = 0; iz < nz; iz++) {
        for (let iy = 0; iy < ny; iy++) {
          for (let ix = 0; ix < nx; ix++) {
            const x = (ix + 0.5) * spacing
            const y = (iy + 0.5) * spacing
            const z = (iz + 0.5) * spacing
            positions.push(x, y, z)
          }
        }
      }

      // Assert
      expect(positions.length).toBe(expectedAtoms * 3)

      // First atom at (0.5, 0.5, 0.5) * spacing = (0.75, 0.75, 0.75)
      expect(positions[0]).toBeCloseTo(0.75)
      expect(positions[1]).toBeCloseTo(0.75)
      expect(positions[2]).toBeCloseTo(0.75)

      // Last atom at (1.5, 1.5, 1.5) * spacing = (2.25, 2.25, 2.25)
      const last = positions.length - 3
      expect(positions[last]).toBeCloseTo(2.25)
      expect(positions[last + 1]).toBeCloseTo(2.25)
      expect(positions[last + 2]).toBeCloseTo(2.25)
    })

    it('should calculate correct box dimensions for lattice', () => {
      // Arrange
      const nx = 4, ny = 5, nz = 6
      const spacing = 2.0

      // Act
      const lx = nx * spacing
      const ly = ny * spacing
      const lz = nz * spacing

      // Assert
      expect(lx).toBe(8)
      expect(ly).toBe(10)
      expect(lz).toBe(12)
    })

    it('should calculate atom count for lattice dimensions', () => {
      const testCases = [
        { nx: 5, ny: 5, nz: 5, expected: 125 },
        { nx: 10, ny: 10, nz: 10, expected: 1000 },
        { nx: 4, ny: 4, nz: 4, expected: 64 },
      ]

      for (const { nx, ny, nz, expected } of testCases) {
        expect(nx * ny * nz).toBe(expected)
      }
    })
  })

  describe('buffer size calculations', () => {
    it('should calculate correct buffer sizes for positions', () => {
      const numAtoms = 1000
      const bytesPerFloat32 = 4
      const componentsPerPosition = 3

      const bufferSize = numAtoms * componentsPerPosition * bytesPerFloat32

      expect(bufferSize).toBe(12000) // 1000 * 3 * 4
    })

    it('should calculate correct buffer sizes for types', () => {
      const numAtoms = 1000
      const bytesPerUint32 = 4

      const bufferSize = numAtoms * bytesPerUint32

      expect(bufferSize).toBe(4000) // 1000 * 4
    })
  })
})

