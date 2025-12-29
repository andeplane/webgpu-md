import { WebGPUContext } from '../compute/WebGPUContext'
import { NeighborList } from '../compute/NeighborList'
import { SimulationBox } from './SimulationBox'
import { SimulationState } from './SimulationState'
import type { PairStyle } from '../pair-styles/PairStyle'
import type { Integrator } from '../integrators/Integrator'
import { VelocityVerlet } from '../integrators/VelocityVerlet'
import { PairLJCut } from '../pair-styles/PairLJCut'
import { parseLAMMPSData } from '../parsers/lammpsdataparser'

/**
 * Configuration for simulation
 */
export interface SimulationConfig {
  /** Number of atoms */
  numAtoms: number
  /** Number of atom types */
  numTypes?: number
  /** Simulation box */
  box?: SimulationBox
  /** Timestep (LJ units, default 0.005) */
  dt?: number
  /** Cutoff distance for pair interactions */
  cutoff?: number
  /** Skin distance for neighbor list */
  skin?: number
  /** Maximum neighbors per atom */
  maxNeighbors?: number
  /** Neighbor list rebuild frequency (steps) */
  neighEvery?: number
}

/**
 * Statistics from simulation
 */
export interface SimulationStats {
  timestep: number
  potentialEnergy?: number
  kineticEnergy?: number
  temperature?: number
}

/**
 * Main simulation class orchestrating MD simulation on WebGPU
 */
export class Simulation {
  readonly ctx: WebGPUContext
  readonly state: SimulationState
  readonly neighborList: NeighborList
  readonly integrator: Integrator
  readonly pairStyle: PairStyle

  private currentStep = 0
  private neighRebuildEvery: number
  private lastNeighRebuild = 0

  private constructor(
    ctx: WebGPUContext,
    state: SimulationState,
    neighborList: NeighborList,
    integrator: Integrator,
    pairStyle: PairStyle,
    neighEvery: number
  ) {
    this.ctx = ctx
    this.state = state
    this.neighborList = neighborList
    this.integrator = integrator
    this.pairStyle = pairStyle
    this.neighRebuildEvery = neighEvery
  }

  /**
   * Create a simulation from configuration
   */
  static async create(config: SimulationConfig): Promise<Simulation> {
    const ctx = await WebGPUContext.create()
    
    const numTypes = config.numTypes ?? 1
    const dt = config.dt ?? 0.005
    const cutoff = config.cutoff ?? 2.5
    const skin = config.skin ?? 0.3
    const maxNeighbors = config.maxNeighbors ?? 128
    const neighEvery = config.neighEvery ?? 10

    // Create state
    const state = new SimulationState({
      device: ctx.device,
      numAtoms: config.numAtoms,
      maxNeighbors,
    })

    // Set box
    if (config.box) {
      state.setBox(config.box)
    }

    // Create neighbor list
    const neighborList = new NeighborList(
      ctx,
      config.numAtoms,
      cutoff,
      skin,
      maxNeighbors
    )
    neighborList.updateBox(state.box)

    // Create integrator
    const integrator = new VelocityVerlet(ctx, config.numAtoms, { dt })
    integrator.updateBox(state)

    // Create pair style
    const pairStyle = new PairLJCut(ctx, config.numAtoms, {
      cutoff,
      numTypes,
    })
    pairStyle.updateBox(state)

    return new Simulation(ctx, state, neighborList, integrator, pairStyle, neighEvery)
  }

  /**
   * Create a simulation from a LAMMPS data file
   */
  static async fromLAMMPSData(
    data: string,
    options: {
      dt?: number
      cutoff?: number
      skin?: number
      maxNeighbors?: number
      neighEvery?: number
    } = {}
  ): Promise<Simulation> {
    const parsed = parseLAMMPSData(data)
    
    const sim = await Simulation.create({
      numAtoms: parsed.numAtoms,
      numTypes: parsed.numTypes,
      box: parsed.box,
      dt: options.dt ?? 0.005,
      cutoff: options.cutoff ?? 2.5,
      skin: options.skin ?? 0.3,
      maxNeighbors: options.maxNeighbors ?? 128,
      neighEvery: options.neighEvery ?? 10,
    })

    // Upload initial data
    sim.state.setPositions(parsed.positions)
    sim.state.setTypes(parsed.types)
    sim.state.setMasses(sim.expandMassesPerType(parsed.masses, parsed.types))

    return sim
  }

  /**
   * Create a simple LJ liquid system using FCC lattice
   * 
   * @param nx - number of FCC unit cells in x direction
   * @param ny - number of FCC unit cells in y direction
   * @param nz - number of FCC unit cells in z direction
   * @param options - simulation options
   * 
   * Note: Total atoms = 4 * nx * ny * nz (FCC has 4 atoms per unit cell)
   */
  static async createLJLiquid(
    nx: number,
    ny: number,
    nz: number,
    options: {
      density?: number
      temperature?: number
      epsilon?: number
      sigma?: number
      dt?: number
      cutoff?: number
    } = {}
  ): Promise<Simulation> {
    // FCC has 4 atoms per unit cell
    const numAtoms = 4 * nx * ny * nz
    const density = options.density ?? 0.8
    const temperature = options.temperature ?? 1.0
    const epsilon = options.epsilon ?? 1.0
    const sigma = options.sigma ?? 1.0
    const dt = options.dt ?? 0.005
    const cutoff = options.cutoff ?? 2.5 * sigma

    // Calculate FCC lattice constant from density
    // For FCC: density = 4 * mass / a^3, so a = (4 / density)^(1/3)
    // In reduced units with mass = 1:
    const latticeConstant = Math.pow(4.0 / density, 1.0 / 3.0)
    
    // Box size
    const lx = nx * latticeConstant
    const ly = ny * latticeConstant
    const lz = nz * latticeConstant
    
    const sim = await Simulation.create({
      numAtoms,
      numTypes: 1,
      box: SimulationBox.fromDimensions(lx, ly, lz),
      dt,
      cutoff,
    })

    // Initialize FCC lattice
    sim.state.initializeFCC(nx, ny, nz, latticeConstant, 0)

    // Set LJ coefficients
    sim.pairStyle.setCoeff(0, 0, epsilon, sigma)

    // Initialize velocities with Maxwell-Boltzmann distribution
    sim.state.initializeVelocities(temperature)

    // Update components for the new box
    sim.neighborList.updateBox(sim.state.box)
    sim.integrator.updateBox(sim.state)
    sim.pairStyle.updateBox(sim.state)

    // Build initial neighbor list
    sim.neighborList.build(sim.state.positionsBuffer)
    sim.lastNeighRebuild = 0

    // Zero initial forces (they'll be computed on first step)
    sim.state.zeroForces()

    return sim
  }

  /**
   * Get the current timestep
   */
  get timestep(): number {
    return this.currentStep
  }

  /**
   * Set pair coefficients
   */
  setCoeff(typeI: number, typeJ: number, ...params: number[]): void {
    this.pairStyle.setCoeff(typeI, typeJ, ...params)
  }

  /**
   * Run simulation for specified number of steps
   */
  run(numSteps: number, options: { logEvery?: number, computeEnergy?: boolean } = {}): void {
    const logEvery = options.logEvery ?? 0
    
    for (let step = 0; step < numSteps; step++) {
      this.step()

      if (logEvery > 0 && this.currentStep % logEvery === 0) {
        console.log(`Step ${this.currentStep}`)
      }
    }
  }

  /**
   * Run a single timestep using velocity Verlet integration
   */
  step(): void {
    // Velocity Verlet integration:
    // 1. v(t+dt/2) = v(t) + 0.5 * dt * f(t) / m
    // 2. x(t+dt) = x(t) + dt * v(t+dt/2)
    this.integrator.integrateInitial(this.state)

    // 3. Rebuild neighbor list if needed (after position update)
    if (this.currentStep - this.lastNeighRebuild >= this.neighRebuildEvery) {
      this.neighborList.build(this.state.positionsBuffer)
      this.lastNeighRebuild = this.currentStep
    }

    // 4. Compute forces f(t+dt)
    this.pairStyle.compute(this.state, this.neighborList)

    // 5. v(t+dt) = v(t+dt/2) + 0.5 * dt * f(t+dt) / m
    this.integrator.integrateFinal(this.state)

    this.currentStep++
  }

  /**
   * Force neighbor list rebuild on next step
   */
  forceNeighborRebuild(): void {
    this.lastNeighRebuild = -this.neighRebuildEvery
  }

  /**
   * Expand per-type masses to per-atom masses
   */
  private expandMassesPerType(massesPerType: Float32Array, types: Uint32Array): Float32Array {
    const masses = new Float32Array(types.length)
    for (let i = 0; i < types.length; i++) {
      masses[i] = massesPerType[types[i]]
    }
    return masses
  }

  /**
   * Read positions back from GPU
   */
  async readPositions(): Promise<Float32Array> {
    return this.state.readPositions()
  }

  /**
   * Read types back from GPU
   */
  async readTypes(): Promise<Uint32Array> {
    return this.state.readTypes()
  }

  /**
   * Get simulation box
   */
  get box(): SimulationBox {
    return this.state.box
  }

  /**
   * Get number of atoms
   */
  get numAtoms(): number {
    return this.state.numAtoms
  }

  /**
   * Destroy GPU resources
   */
  destroy(): void {
    this.state.destroy()
    this.neighborList.destroy()
    this.integrator.destroy()
    this.pairStyle.destroy()
    this.ctx.destroy()
  }
}

