import { describe, it, expect } from 'vitest'
import { PairCoeffMatrix } from './PairStyle'

interface LJCoeff {
  epsilon: number
  sigma: number
  cutoff: number
  [key: string]: number
}

describe('PairCoeffMatrix', () => {
  describe('basic operations', () => {
    it('should store and retrieve coefficients', () => {
      // Arrange
      const matrix = new PairCoeffMatrix<LJCoeff>(3)
      const coeffs: LJCoeff = { epsilon: 1.0, sigma: 1.0, cutoff: 2.5 }

      // Act
      matrix.set(0, 1, coeffs)
      const result = matrix.get(0, 1)

      // Assert
      expect(result).toEqual(coeffs)
    })

    it('should retrieve symmetric coefficients', () => {
      // Arrange
      const matrix = new PairCoeffMatrix<LJCoeff>(3)
      const coeffs: LJCoeff = { epsilon: 1.5, sigma: 3.4, cutoff: 10.0 }

      // Act
      matrix.set(0, 2, coeffs)

      // Assert - should work both ways
      expect(matrix.get(0, 2)).toEqual(coeffs)
      expect(matrix.get(2, 0)).toEqual(coeffs)
    })

    it('should check existence with has()', () => {
      // Arrange
      const matrix = new PairCoeffMatrix<LJCoeff>(2)

      // Act
      matrix.set(0, 0, { epsilon: 1.0, sigma: 1.0, cutoff: 2.5 })

      // Assert
      expect(matrix.has(0, 0)).toBe(true)
      expect(matrix.has(0, 1)).toBe(false)
      expect(matrix.has(1, 1)).toBe(false)
    })
  })

  describe('entries()', () => {
    it('should return all set entries', () => {
      // Arrange
      const matrix = new PairCoeffMatrix<LJCoeff>(2)
      matrix.set(0, 0, { epsilon: 1.0, sigma: 1.0, cutoff: 2.5 })
      matrix.set(0, 1, { epsilon: 1.5, sigma: 1.2, cutoff: 3.0 })
      matrix.set(1, 1, { epsilon: 0.5, sigma: 3.0, cutoff: 7.5 })

      // Act
      const entries = matrix.entries()

      // Assert
      expect(entries.length).toBe(3)
      
      const entry01 = entries.find(([i, j]) => i === 0 && j === 1)
      expect(entry01).toBeDefined()
      expect(entry01![2].epsilon).toBe(1.5)
    })
  })

  describe('packForGPU', () => {
    it('should pack coefficients into GPU-friendly format', () => {
      // Arrange
      const matrix = new PairCoeffMatrix<LJCoeff>(2)
      matrix.set(0, 0, { epsilon: 1.0, sigma: 1.0, cutoff: 2.5 })
      matrix.set(0, 1, { epsilon: 1.5, sigma: 1.2, cutoff: 3.0 })
      matrix.set(1, 1, { epsilon: 0.5, sigma: 3.0, cutoff: 7.5 })

      const packer = (c: LJCoeff) => [c.epsilon, c.sigma, c.cutoff, 0]

      // Act
      const data = matrix.packForGPU(4, packer)

      // Assert - should be 2x2x4 = 16 elements
      expect(data.length).toBe(16)
      expect(data).toBeInstanceOf(Float32Array)

      // Index [0,0] at offset 0
      // Use toBeCloseTo for Float32 precision
      expect(data[0]).toBeCloseTo(1.0, 5) // epsilon
      expect(data[1]).toBeCloseTo(1.0, 5) // sigma
      expect(data[2]).toBeCloseTo(2.5, 5) // cutoff

      // Index [0,1] at offset 4
      expect(data[4]).toBeCloseTo(1.5, 5) // epsilon
      expect(data[5]).toBeCloseTo(1.2, 5) // sigma
      expect(data[6]).toBeCloseTo(3.0, 5) // cutoff

      // Index [1,0] at offset 8 - should be symmetric
      expect(data[8]).toBeCloseTo(1.5, 5) // epsilon (same as [0,1])
      expect(data[9]).toBeCloseTo(1.2, 5) // sigma
      expect(data[10]).toBeCloseTo(3.0, 5) // cutoff

      // Index [1,1] at offset 12
      expect(data[12]).toBeCloseTo(0.5, 5) // epsilon
      expect(data[13]).toBeCloseTo(3.0, 5) // sigma
      expect(data[14]).toBeCloseTo(7.5, 5) // cutoff
    })

    it('should handle larger type matrices', () => {
      // Arrange
      const matrix = new PairCoeffMatrix<LJCoeff>(3)
      // Only set diagonal
      for (let i = 0; i < 3; i++) {
        matrix.set(i, i, { epsilon: i + 1.0, sigma: 1.0, cutoff: 2.5 })
      }

      const packer = (c: LJCoeff) => [c.epsilon, c.sigma]

      // Act
      const data = matrix.packForGPU(2, packer)

      // Assert - should be 3x3x2 = 18 elements
      expect(data.length).toBe(18)

      // Check diagonal entries
      expect(data[0]).toBe(1.0)  // [0,0] epsilon
      expect(data[8]).toBe(2.0)  // [1,1] epsilon = index (1*3+1)*2 = 8
      expect(data[16]).toBe(3.0) // [2,2] epsilon = index (2*3+2)*2 = 16
    })

    it('should initialize unset entries to zero', () => {
      // Arrange
      const matrix = new PairCoeffMatrix<LJCoeff>(2)
      // Only set one entry
      matrix.set(0, 0, { epsilon: 1.0, sigma: 1.0, cutoff: 2.5 })

      const packer = (c: LJCoeff) => [c.epsilon, c.sigma, c.cutoff, 0]

      // Act
      const data = matrix.packForGPU(4, packer)

      // Assert - unset entries should be 0
      // [1,1] at offset 12 should be 0
      expect(data[12]).toBe(0)
      expect(data[13]).toBe(0)
      expect(data[14]).toBe(0)
    })
  })
})

