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
 * Starts at 4,000 atoms and increases by 25% each time up to ~4M atoms
 */
function getSystemSizes() {
  const sizes: { nx: number, ny: number, nz: number }[] = []
  let currentAtoms = 4000
  const maxAtoms = 4000000
  const multiplier = 1.25

  while (currentAtoms <= maxAtoms * 1.1) { // Allow slightly over 4M for the last step
    const n = Math.round(Math.pow(currentAtoms / 4, 1/3))
    
    // Ensure we don't add the same size twice due to rounding
    if (sizes.length === 0 || sizes[sizes.length - 1].nx !== n) {
      sizes.push({ nx: n, ny: n, nz: n })
    }
    
    currentAtoms *= multiplier
  }
  return sizes
}

const SYSTEM_SIZES = getSystemSizes()

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

