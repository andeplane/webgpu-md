import type { Simulation } from './Simulation'
import type { ProfileStats } from './SimulationProfiler'

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
  /** Profiling breakdown (if profiling was enabled) */
  profiling?: ProfileStats
}

/**
 * Benchmark configuration
 */
export interface BenchmarkConfig {
  /** Number of warmup steps (not counted) */
  warmupSteps?: number
  /** Number of benchmark steps */
  benchmarkSteps?: number
  /** Enable detailed profiling (slower but provides timing breakdown) */
  profile?: boolean
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
  const profile = config.profile ?? false

  console.log(`Benchmark: ${simulation.numAtoms} atoms`)
  console.log(`Warmup: ${warmupSteps} steps, Benchmark: ${benchmarkSteps} steps`)
  if (profile) {
    console.log('Profiling enabled (slower but provides timing breakdown)')
  }

  // Warmup phase - uses the SAME run() method
  console.log('Running warmup...')
  await simulation.run(warmupSteps)
  await simulation.ctx.waitForGPU()

  // Enable profiling if requested
  let profiler = null
  if (profile) {
    profiler = simulation.enableProfiling()
  }

  // Benchmark phase
  console.log('Running benchmark...')
  const startTime = performance.now()

  if (profile) {
    // Use profiling version (slower due to GPU sync)
    await simulation.runWithProfiling(benchmarkSteps)
  } else {
    // Use normal fast version
    await simulation.run(benchmarkSteps)
  }

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

  // Add profiling stats if enabled
  if (profiler) {
    result.profiling = profiler.getStats()
    console.log('')
    console.log(profiler.formatStats())
  }

  // Disable profiling
  simulation.disableProfiling()

  console.log('')
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