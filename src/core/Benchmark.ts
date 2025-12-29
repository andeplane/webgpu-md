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
  /** Callback for progress updates */
  onProgress?: (step: number, totalSteps: number) => void
}

/**
 * Run a benchmark on a simulation without visualization
 * Measures raw simulation performance (forces, neighbor lists, integration)
 */
export async function runBenchmark(
  simulation: Simulation,
  config: BenchmarkConfig = {}
): Promise<BenchmarkResult> {
  const warmupSteps = config.warmupSteps ?? 100
  const benchmarkSteps = config.benchmarkSteps ?? 1000
  const onProgress = config.onProgress

  console.log(`Benchmark: ${simulation.numAtoms} atoms`)
  console.log(`Warmup: ${warmupSteps} steps, Benchmark: ${benchmarkSteps} steps`)

  // Warmup phase - let GPU compile shaders and warm up caches
  console.log('Running warmup...')
  for (let i = 0; i < warmupSteps; i++) {
    simulation.step()
  }

  // Wait for GPU to finish warmup
  await simulation.ctx.waitForGPU()

  // Benchmark phase
  console.log('Running benchmark...')
  const startTime = performance.now()

  for (let i = 0; i < benchmarkSteps; i++) {
    simulation.step()
    
    if (onProgress && i % 100 === 0) {
      onProgress(i, benchmarkSteps)
    }
  }

  // Wait for all GPU work to complete
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

/**
 * Run a series of benchmarks with increasing system sizes
 */
export async function runScalingBenchmark(
  sizes: number[],
  config: BenchmarkConfig & {
    density?: number
    temperature?: number
  } = {}
): Promise<BenchmarkResult[]> {
  const { Simulation } = await import('./Simulation')
  const results: BenchmarkResult[] = []

  for (const n of sizes) {
    // Create cubic system with n^3 atoms
    const simulation = await Simulation.createLJLiquid(n, n, n, {
      density: config.density ?? 0.8,
      temperature: config.temperature ?? 1.0,
    })

    // Set LJ coefficients
    simulation.setCoeff(0, 0, 1.0, 1.0)

    // Run benchmark
    const result = await runBenchmark(simulation, config)
    results.push(result)

    // Cleanup
    simulation.destroy()
  }

  // Print scaling summary
  console.log('\n=== Scaling Summary ===')
  console.log('Atoms\t\tM atom-steps/s\tns/atom-step')
  for (const r of results) {
    console.log(`${r.numAtoms}\t\t${r.millionAtomStepsPerSecond.toFixed(2)}\t\t${r.nsPerAtomStep.toFixed(2)}`)
  }
  console.log('=======================\n')

  return results
}

