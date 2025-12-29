import type { Simulation } from './Simulation'

/**
 * Benchmark result
 */
export interface BenchmarkResult {
  /** Total number of atoms */
  numAtoms: number
  /** Total steps run */
  totalSteps: number
  /** Total time in seconds */
  totalTimeSeconds: number
  /** Steps per second */
  stepsPerSecond: number
  /** Million atom-steps per second */
  millionAtomStepsPerSecond: number
  /** Nanoseconds per atom-step */
  nsPerAtomStep: number
}

/**
 * Benchmark configuration
 */
export interface BenchmarkConfig {
  /** Number of warmup steps (not counted) */
  warmupSteps?: number
  /** Number of benchmark steps */
  benchmarkSteps?: number
}

/**
 * Run a benchmark on a simulation without visualization
 * Uses the exact same simulation.run() method as normal execution
 */
export async function runBenchmark(
  simulation: Simulation,
  config: BenchmarkConfig = {}
): Promise<BenchmarkResult> {
  const warmupSteps = config.warmupSteps ?? 100
  const benchmarkSteps = config.benchmarkSteps ?? 1000

  console.log(`Benchmark: ${simulation.numAtoms} atoms`)
  console.log(`Warmup: ${warmupSteps} steps, Benchmark: ${benchmarkSteps} steps`)

  // Warmup phase - uses the SAME run() method
  console.log('Running warmup...')
  simulation.run(warmupSteps)
  await simulation.ctx.waitForGPU()

  // Benchmark phase - uses the SAME run() method
  console.log('Running benchmark...')
  const startTime = performance.now()

  simulation.run(benchmarkSteps)

  await simulation.ctx.waitForGPU()
  const endTime = performance.now()

  const totalTimeSeconds = (endTime - startTime) / 1000

  // Calculate metrics
  const stepsPerSecond = benchmarkSteps / totalTimeSeconds
  const atomStepsPerSecond = stepsPerSecond * simulation.numAtoms
  const millionAtomStepsPerSecond = atomStepsPerSecond / 1e6
  const nsPerAtomStep = (totalTimeSeconds * 1e9) / (benchmarkSteps * simulation.numAtoms)

  const result: BenchmarkResult = {
    numAtoms: simulation.numAtoms,
    totalSteps: benchmarkSteps,
    totalTimeSeconds,
    stepsPerSecond,
    millionAtomStepsPerSecond,
    nsPerAtomStep,
  }

  console.log('--- Benchmark Results ---')
  console.log(`Atoms: ${result.numAtoms}`)
  console.log(`Steps: ${result.totalSteps}`)
  console.log(`Time: ${result.totalTimeSeconds.toFixed(3)} s`)
  console.log(`Steps/sec: ${result.stepsPerSecond.toFixed(1)}`)
  console.log(`M atom-steps/sec: ${result.millionAtomStepsPerSecond.toFixed(2)}`)
  console.log(`ns/atom-step: ${result.nsPerAtomStep.toFixed(2)}`)
  console.log('-------------------------')

  return result
}
