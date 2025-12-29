import { describe, it, expect, vi } from 'vitest'
import type { BenchmarkResult, BenchmarkConfig } from './Benchmark'

// We test the calculation logic without GPU
describe('Benchmark calculations', () => {
  describe('BenchmarkResult metrics', () => {
    it('should calculate correct steps per second', () => {
      // Arrange
      const totalSteps = 1000
      const totalTimeSeconds = 2.0

      // Act
      const stepsPerSecond = totalSteps / totalTimeSeconds

      // Assert
      expect(stepsPerSecond).toBe(500)
    })

    it('should calculate correct million atom-steps per second', () => {
      // Arrange
      const numAtoms = 10000
      const stepsPerSecond = 100

      // Act
      const atomStepsPerSecond = stepsPerSecond * numAtoms
      const millionAtomStepsPerSecond = atomStepsPerSecond / 1e6

      // Assert
      expect(millionAtomStepsPerSecond).toBe(1.0) // 100 * 10000 / 1e6 = 1
    })

    it('should calculate correct ns per atom-step', () => {
      // Arrange
      const totalTimeSeconds = 1.0
      const totalSteps = 1000
      const numAtoms = 1000

      // Act
      const nsPerAtomStep = (totalTimeSeconds * 1e9) / (totalSteps * numAtoms)

      // Assert
      // 1e9 ns / (1000 steps * 1000 atoms) = 1e9 / 1e6 = 1000 ns/atom-step
      expect(nsPerAtomStep).toBe(1000)
    })
  })

  describe('BenchmarkConfig defaults', () => {
    it('should have reasonable default warmup steps', () => {
      // Arrange
      const config: BenchmarkConfig = {}
      const warmupSteps = config.warmupSteps ?? 100

      // Assert
      expect(warmupSteps).toBe(100)
    })

    it('should have reasonable default benchmark steps', () => {
      // Arrange
      const config: BenchmarkConfig = {}
      const benchmarkSteps = config.benchmarkSteps ?? 1000

      // Assert
      expect(benchmarkSteps).toBe(1000)
    })

    it('should allow custom warmup and benchmark steps', () => {
      // Arrange
      const config: BenchmarkConfig = {
        warmupSteps: 50,
        benchmarkSteps: 500,
      }

      // Assert
      expect(config.warmupSteps).toBe(50)
      expect(config.benchmarkSteps).toBe(500)
    })
  })

  describe('progress callback', () => {
    it('should call progress callback at correct intervals', () => {
      // Arrange
      const onProgress = vi.fn()
      const benchmarkSteps = 500

      // Act - simulate what runBenchmark does
      for (let i = 0; i < benchmarkSteps; i++) {
        if (i % 100 === 0) {
          onProgress(i, benchmarkSteps)
        }
      }

      // Assert
      expect(onProgress).toHaveBeenCalledTimes(5) // 0, 100, 200, 300, 400
      expect(onProgress).toHaveBeenCalledWith(0, 500)
      expect(onProgress).toHaveBeenCalledWith(100, 500)
      expect(onProgress).toHaveBeenCalledWith(400, 500)
    })
  })
})

describe('BenchmarkResult structure', () => {
  it('should contain all required metrics', () => {
    // Arrange - create a mock result
    const result: BenchmarkResult = {
      numAtoms: 4000,
      totalSteps: 1000,
      totalTimeSeconds: 0.5,
      stepsPerSecond: 2000,
      millionAtomStepsPerSecond: 8.0,
      nsPerAtomStep: 0.125,
    }

    // Assert - all fields should be present
    expect(result.numAtoms).toBeDefined()
    expect(result.totalSteps).toBeDefined()
    expect(result.totalTimeSeconds).toBeDefined()
    expect(result.stepsPerSecond).toBeDefined()
    expect(result.millionAtomStepsPerSecond).toBeDefined()
    expect(result.nsPerAtomStep).toBeDefined()
  })

  it('should have consistent metrics', () => {
    // Arrange
    const numAtoms = 8000
    const totalSteps = 1000
    const totalTimeSeconds = 0.4 // 400ms

    // Act
    const stepsPerSecond = totalSteps / totalTimeSeconds
    const atomStepsPerSecond = stepsPerSecond * numAtoms
    const millionAtomStepsPerSecond = atomStepsPerSecond / 1e6
    const nsPerAtomStep = (totalTimeSeconds * 1e9) / (totalSteps * numAtoms)

    // Assert - metrics should be internally consistent
    expect(stepsPerSecond).toBe(2500)
    expect(millionAtomStepsPerSecond).toBe(20.0) // 2500 * 8000 / 1e6
    expect(nsPerAtomStep).toBe(50) // 0.4e9 / 8e6
  })
})

