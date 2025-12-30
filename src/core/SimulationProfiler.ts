/**
 * Profiling statistics for simulation performance breakdown
 */
export interface ProfileStats {
  /** Time spent in force calculation (ms) */
  forceCalculation: number
  /** Time spent in integration (ms) */
  integration: number
  /** Time spent building neighbor lists (ms) */
  neighborListBuild: number
  /** Time spent copying data to CPU (ms, visualization only) */
  dataCopy: number
  /** Time spent in other operations (ms) */
  other: number
  /** Total time (ms) */
  total: number
  /** Number of steps profiled */
  stepCount: number
}

/**
 * Profiler for measuring simulation performance breakdown
 * 
 * Usage:
 * ```
 * const profiler = new SimulationProfiler()
 * profiler.start('forceCalculation')
 * // ... do force calculation ...
 * await profiler.end('forceCalculation', ctx)
 * ```
 */
export class SimulationProfiler {
  private timings: Map<keyof ProfileStats, number> = new Map()
  private startTimes: Map<string, number> = new Map()
  private stepCount = 0

  constructor() {
    this.reset()
  }

  /**
   * Reset all accumulated timings
   */
  reset(): void {
    this.timings.set('forceCalculation', 0)
    this.timings.set('integration', 0)
    this.timings.set('neighborListBuild', 0)
    this.timings.set('dataCopy', 0)
    this.timings.set('other', 0)
    this.timings.set('total', 0)
    this.stepCount = 0
  }

  /**
   * Start timing a category
   */
  start(category: string): void {
    this.startTimes.set(category, performance.now())
  }

  /**
   * End timing a category and accumulate the time
   * @param category - The category to end timing for
   * @param waitForGPU - Optional async function to wait for GPU completion
   */
  async end(category: keyof ProfileStats, waitForGPU?: () => Promise<void>): Promise<void> {
    // Wait for GPU to complete if provided (for accurate timing)
    if (waitForGPU) {
      await waitForGPU()
    }

    const startTime = this.startTimes.get(category)
    if (startTime !== undefined) {
      const elapsed = performance.now() - startTime
      const current = this.timings.get(category) ?? 0
      this.timings.set(category, current + elapsed)
      this.startTimes.delete(category)
    }
  }

  /**
   * End timing synchronously (when GPU sync is handled externally)
   */
  endSync(category: keyof ProfileStats): void {
    const startTime = this.startTimes.get(category)
    if (startTime !== undefined) {
      const elapsed = performance.now() - startTime
      const current = this.timings.get(category) ?? 0
      this.timings.set(category, current + elapsed)
      this.startTimes.delete(category)
    }
  }

  /**
   * Mark the end of a simulation step
   */
  endStep(): void {
    this.stepCount++
  }

  /**
   * Get accumulated statistics
   */
  getStats(): ProfileStats {
    const forceCalculation = this.timings.get('forceCalculation') ?? 0
    const integration = this.timings.get('integration') ?? 0
    const neighborListBuild = this.timings.get('neighborListBuild') ?? 0
    const dataCopy = this.timings.get('dataCopy') ?? 0
    const other = this.timings.get('other') ?? 0
    const total = forceCalculation + integration + neighborListBuild + dataCopy + other

    return {
      forceCalculation,
      integration,
      neighborListBuild,
      dataCopy,
      other,
      total,
      stepCount: this.stepCount,
    }
  }

  /**
   * Get per-step average statistics
   */
  getPerStepStats(): ProfileStats {
    const stats = this.getStats()
    if (this.stepCount === 0) return stats

    return {
      forceCalculation: stats.forceCalculation / this.stepCount,
      integration: stats.integration / this.stepCount,
      neighborListBuild: stats.neighborListBuild / this.stepCount,
      dataCopy: stats.dataCopy / this.stepCount,
      other: stats.other / this.stepCount,
      total: stats.total / this.stepCount,
      stepCount: this.stepCount,
    }
  }

  /**
   * Format stats as a human-readable string
   */
  formatStats(): string {
    const stats = this.getStats()
    const perStep = this.getPerStepStats()
    
    if (stats.total === 0) return 'No profiling data'

    const pct = (val: number) => ((val / stats.total) * 100).toFixed(1)
    const ms = (val: number) => val.toFixed(2)

    const lines = [
      '--- Timing Breakdown ---',
      `Force calculation:  ${pct(stats.forceCalculation).padStart(5)}% (${ms(perStep.forceCalculation)} ms/step)`,
      `Neighbor list:      ${pct(stats.neighborListBuild).padStart(5)}% (${ms(perStep.neighborListBuild)} ms/step)`,
      `Integration:        ${pct(stats.integration).padStart(5)}% (${ms(perStep.integration)} ms/step)`,
    ]

    if (stats.dataCopy > 0) {
      lines.push(`Data copy:          ${pct(stats.dataCopy).padStart(5)}% (${ms(perStep.dataCopy)} ms/step)`)
    }

    if (stats.other > 0) {
      lines.push(`Other:              ${pct(stats.other).padStart(5)}% (${ms(perStep.other)} ms/step)`)
    }

    lines.push(`------------------------`)
    lines.push(`Total: ${ms(stats.total)} ms (${this.stepCount} steps, ${ms(perStep.total)} ms/step)`)

    return lines.join('\n')
  }
}

