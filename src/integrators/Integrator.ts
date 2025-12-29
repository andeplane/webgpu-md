import type { WebGPUContext } from '../compute/WebGPUContext'
import type { SimulationState } from '../core/SimulationState'

/**
 * Configuration for an integrator
 */
export interface IntegratorConfig {
  /** Timestep */
  dt: number
}

/**
 * Abstract base class for time integrators
 */
export abstract class Integrator {
  protected ctx: WebGPUContext
  protected numAtoms: number
  protected dt: number

  /** Name of the integrator */
  abstract readonly name: string

  constructor(ctx: WebGPUContext, numAtoms: number, config: IntegratorConfig) {
    this.ctx = ctx
    this.numAtoms = numAtoms
    this.dt = config.dt
  }

  /**
   * Get the timestep
   */
  getTimestep(): number {
    return this.dt
  }

  /**
   * Set the timestep
   */
  setTimestep(dt: number): void {
    this.dt = dt
    this.updateParams()
  }

  /**
   * Update parameter buffer after timestep change
   */
  protected abstract updateParams(): void

  /**
   * Update for new simulation box
   */
  abstract updateBox(state: SimulationState): void

  /**
   * Initial integration phase (before force computation)
   * Typically: v += 0.5*dt*f/m, x += dt*v
   */
  abstract integrateInitial(state: SimulationState): void

  /**
   * Final integration phase (after force computation)
   * Typically: v += 0.5*dt*f/m
   */
  abstract integrateFinal(state: SimulationState): void

  /**
   * Clean up GPU resources
   */
  abstract destroy(): void
}

