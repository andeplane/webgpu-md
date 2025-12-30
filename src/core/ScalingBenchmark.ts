import { Simulation } from './Simulation'
import { runBenchmark, type BenchmarkResult } from './Benchmark'

/**
 * Result for a single system size in the scaling benchmark
 */
export interface ScalingResult {
  unitCells: number
  numAtoms: number
  stepsPerSec: number
  mAtomStepsPerSec: number
  // Optional profiling breakdown (when enabled)
  profiling?: {
    forceCalcPercent: number
    neighborListPercent: number
    integrationPercent: number
  }
}

/**
 * System sizes for scaling benchmark
 * FCC unit cell sizes providing roughly 256k increments up to 4M atoms
 */
const SYSTEM_SIZES = [
  { nx: 10, ny: 10, nz: 10 }, // 4,000 atoms
  { nx: 40, ny: 40, nz: 40 }, // 256,000 atoms
  { nx: 50, ny: 50, nz: 50 }, // 500,000 atoms
  { nx: 58, ny: 58, nz: 58 }, // 780,448 atoms
  { nx: 65, ny: 65, nz: 65 }, // 1,098,500 atoms
  { nx: 72, ny: 72, nz: 72 }, // 1,492,992 atoms
  { nx: 80, ny: 80, nz: 80 }, // 2,048,000 atoms
  { nx: 87, ny: 87, nz: 87 }, // 2,628,072 atoms
  { nx: 93, ny: 93, nz: 93 }, // 3,214,108 atoms
  { nx: 100, ny: 100, nz: 100 }, // 4,000,000 atoms
]

/**
 * Run scaling benchmark across multiple system sizes
 */
export async function runScalingBenchmark(
  options: { profile: boolean },
  onProgress: (result: ScalingResult, index: number, total: number) => void
): Promise<ScalingResult[]> {
  const results: ScalingResult[] = []
  const total = SYSTEM_SIZES.length

  for (let i = 0; i < SYSTEM_SIZES.length; i++) {
    const { nx, ny, nz } = SYSTEM_SIZES[i]
    const numAtoms = 4 * nx * ny * nz // FCC has 4 atoms per unit cell

    console.log(`\n[${i + 1}/${total}] Running benchmark for ${nx}×${nx}×${nx} unit cells (${numAtoms.toLocaleString()} atoms)...`)

    let simulation: Simulation | undefined
    try {
      // Create simulation
      simulation = await Simulation.createLJLiquid(nx, ny, nz, {
        density: 0.8,
        temperature: 1.0,
        epsilon: 1.0,
        sigma: 1.0,
        dt: 0.005,
        cutoff: 2.5,
      })

      // Run benchmark
      const benchmarkResult: BenchmarkResult = await runBenchmark(simulation, {
        warmupSteps: 100,
        benchmarkSteps: 1000,
        profile: options.profile,
      })

      // Convert to ScalingResult
      const scalingResult: ScalingResult = {
        unitCells: nx,
        numAtoms,
        stepsPerSec: benchmarkResult.stepsPerSecond,
        mAtomStepsPerSec: benchmarkResult.millionAtomStepsPerSecond,
      }

      // Add profiling breakdown if available
      if (benchmarkResult.profiling && benchmarkResult.profiling.total > 0) {
        const p = benchmarkResult.profiling
        scalingResult.profiling = {
          forceCalcPercent: (p.forceCalculation / p.total) * 100,
          neighborListPercent: (p.neighborListBuild / p.total) * 100,
          integrationPercent: (p.integration / p.total) * 100,
        }
      }

      results.push(scalingResult)
      onProgress(scalingResult, i, total)

    } catch (error) {
      console.error(`Error benchmarking ${nx}×${nx}×${nx}:`, error)
      // Continue with next size even if one fails
    } finally {
      // Clean up simulation
      await simulation?.destroy()
    }
  }

  return results
}

