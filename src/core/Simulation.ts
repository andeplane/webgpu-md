import { WebGPUContext } from '../compute/WebGPUContext'
import { NeighborList } from '../compute/NeighborList'
import { KineticEnergy } from '../compute/KineticEnergy'
import { SimulationBox } from './SimulationBox'
import { SimulationState } from './SimulationState'
import { SimulationProfiler } from './SimulationProfiler'
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
  readonly kineticEnergy: KineticEnergy
  readonly pairStyle: PairStyle

  private currentStep = 0
  private neighRebuildEvery: number
  private lastNeighRebuild = 0
  private skin: number
  private useDisplacementRebuild = true
  private displacementCheckEvery = 10  // Check displacement every N steps

  /** Optional profiler for performance measurement */
  profiler: SimulationProfiler | null = null

  private constructor(
    ctx: WebGPUContext,
    state: SimulationState,
    neighborList: NeighborList,
    integrator: Integrator,
    pairStyle: PairStyle,
    kineticEnergy: KineticEnergy,
    neighEvery: number,
    skin: number
  ) {
    this.ctx = ctx
    this.state = state
    this.neighborList = neighborList
    this.integrator = integrator
    this.pairStyle = pairStyle
    this.kineticEnergy = kineticEnergy
    this.neighRebuildEvery = neighEvery
    this.skin = skin
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

    // Create kinetic energy calculator
    const kineticEnergyCalc = new KineticEnergy(ctx, config.numAtoms)

    return new Simulation(ctx, state, neighborList, integrator, pairStyle, kineticEnergyCalc, neighEvery, skin)
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
    let cutoff = options.cutoff ?? 2.5 * sigma

    // Calculate FCC lattice constant from density
    // For FCC: density = 4 * mass / a^3, so a = (4 / density)^(1/3)
    // In reduced units with mass = 1:
    const latticeConstant = Math.pow(4.0 / density, 1.0 / 3.0)
    
    // Box size
    const lx = nx * latticeConstant
    const ly = ny * latticeConstant
    const lz = nz * latticeConstant
    
    // Check if cutoff is valid for this box size
    // Minimum image convention requires cutoff <= box/2
    const minBox = Math.min(lx, ly, lz)
    const nnDistance = latticeConstant / Math.sqrt(2)  // FCC nearest neighbor distance
    const minRequiredCells = Math.ceil(2 * cutoff / latticeConstant)
    
    if (cutoff > minBox / 2) {
      console.warn(`⚠️ SYSTEM TOO SMALL for cutoff ${cutoff}!`)
      console.warn(`  Box: ${minBox.toFixed(3)}, Box/2: ${(minBox/2).toFixed(3)}`)
      console.warn(`  FCC nearest neighbor distance: ${nnDistance.toFixed(3)}`)
      console.warn(`  Need at least ${minRequiredCells} unit cells for cutoff ${cutoff}`)
      
      // Check if any reasonable cutoff is possible
      const maxSafeCutoff = minBox / 2 - 0.01
      if (maxSafeCutoff < nnDistance) {
        console.error(`❌ FATAL: Box too small! Max safe cutoff (${maxSafeCutoff.toFixed(3)}) < NN distance (${nnDistance.toFixed(3)})`)
        console.error(`   Atoms won't interact! Use at least ${minRequiredCells} unit cells.`)
        throw new Error(`System too small: need at least ${minRequiredCells} unit cells for LJ with cutoff ${cutoff}`)
      }
      
      // Auto-adjust cutoff to be safe
      console.warn(`Auto-adjusting cutoff to ${maxSafeCutoff.toFixed(3)}`)
      cutoff = maxSafeCutoff
    }
    
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

    // Initialize displacement tracking (save positions at last rebuild)
    if (sim.integrator instanceof VelocityVerlet) {
      sim.integrator.savePositionsForRebuild(sim.state)
    }

    // Compute initial forces f(t=0) - needed for velocity Verlet first half-step
    sim.pairStyle.compute(sim.state, sim.neighborList)

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
  run(numSteps: number, options: { logEvery?: number } = {}): void {
    const logEvery = options.logEvery ?? 0
    
    for (let step = 0; step < numSteps; step++) {
      this.step()

      if (logEvery > 0 && this.currentStep % logEvery === 0) {
        console.log(`Step ${this.currentStep}`)
      }
    }
  }

  /**
   * Run simulation for specified number of steps with profiling
   * Use this when you need timing breakdown (slower due to GPU sync)
   */
  async runWithProfiling(numSteps: number, options: { logEvery?: number } = {}): Promise<void> {
    const logEvery = options.logEvery ?? 0
    
    for (let step = 0; step < numSteps; step++) {
      await this.stepWithProfiling()

      if (logEvery > 0 && this.currentStep % logEvery === 0) {
        console.log(`Step ${this.currentStep}`)
      }
    }
  }

  /**
   * Run a single timestep using velocity Verlet integration
   * Uses fixed-interval neighbor list rebuild (for synchronous execution)
   */
  step(): void {
    // Velocity Verlet integration:
    // 1. v(t+dt/2) = v(t) + 0.5 * dt * f(t) / m
    // 2. x(t+dt) = x(t) + dt * v(t+dt/2)
    this.integrator.integrateInitial(this.state)

    // 3. Rebuild neighbor list if needed (after position update)
    // Using fixed interval for synchronous step()
    if (this.currentStep - this.lastNeighRebuild >= this.neighRebuildEvery) {
      this.rebuildNeighborList()
    }

    // 4. Compute forces f(t+dt)
    this.pairStyle.compute(this.state, this.neighborList)

    // 5. v(t+dt) = v(t+dt/2) + 0.5 * dt * f(t+dt) / m
    this.integrator.integrateFinal(this.state)

    this.currentStep++
  }

  /**
   * Run a single timestep with displacement-based neighbor list rebuild (async)
   * More efficient than step() as it only rebuilds when atoms have moved enough
   */
  async stepAsync(): Promise<void> {
    // Velocity Verlet integration:
    // 1. v(t+dt/2) = v(t) + 0.5 * dt * f(t) / m
    // 2. x(t+dt) = x(t) + dt * v(t+dt/2)
    this.integrator.integrateInitial(this.state)

    // 3. Check displacement and rebuild neighbor list if needed
    if (this.useDisplacementRebuild && this.integrator instanceof VelocityVerlet) {
      // Check every N steps to amortize GPU readback cost
      if ((this.currentStep - this.lastNeighRebuild) % this.displacementCheckEvery === 0) {
        const needsRebuild = await this.integrator.needsNeighborRebuild(this.skin)
        if (needsRebuild) {
          this.rebuildNeighborList()
        }
      }
      // Fallback: always rebuild at max interval
      else if (this.currentStep - this.lastNeighRebuild >= this.neighRebuildEvery * 5) {
        this.rebuildNeighborList()
      }
    } else {
      // Fall back to fixed interval
      if (this.currentStep - this.lastNeighRebuild >= this.neighRebuildEvery) {
        this.rebuildNeighborList()
      }
    }

    // 4. Compute forces f(t+dt)
    this.pairStyle.compute(this.state, this.neighborList)

    // 5. v(t+dt) = v(t+dt/2) + 0.5 * dt * f(t+dt) / m
    this.integrator.integrateFinal(this.state)

    this.currentStep++
  }

  /**
   * Rebuild the neighbor list and update displacement tracking
   */
  private rebuildNeighborList(): void {
    this.neighborList.build(this.state.positionsBuffer)
    this.lastNeighRebuild = this.currentStep
    
    // Save positions for displacement tracking
    if (this.integrator instanceof VelocityVerlet) {
      this.integrator.savePositionsForRebuild(this.state)
    }
  }

  /**
   * Run a single timestep with profiling (async for GPU synchronization)
   * This is slower than step() due to GPU sync points, use only for profiling
   * Uses displacement-based neighbor list rebuild
   */
  async stepWithProfiling(): Promise<void> {
    const p = this.profiler
    const waitForGPU = () => this.ctx.waitForGPU()

    // 1. Initial integration (half-step velocity + position update)
    p?.start('integration')
    this.integrator.integrateInitial(this.state)
    if (p) await p.end('integration', waitForGPU)

    // 2. Check displacement and rebuild neighbor list if needed
    let didRebuild = false
    if (this.useDisplacementRebuild && this.integrator instanceof VelocityVerlet) {
      // Check every N steps to amortize GPU readback cost
      if ((this.currentStep - this.lastNeighRebuild) % this.displacementCheckEvery === 0) {
        const needsRebuild = await this.integrator.needsNeighborRebuild(this.skin)
        if (needsRebuild) {
          p?.start('neighborListBuild')
          this.rebuildNeighborList()
          if (p) await p.end('neighborListBuild', waitForGPU)
          didRebuild = true
        }
      }
      // Fallback: always rebuild at max interval
      if (!didRebuild && this.currentStep - this.lastNeighRebuild >= this.neighRebuildEvery * 5) {
        p?.start('neighborListBuild')
        this.rebuildNeighborList()
        if (p) await p.end('neighborListBuild', waitForGPU)
      }
    } else {
      // Fall back to fixed interval
      if (this.currentStep - this.lastNeighRebuild >= this.neighRebuildEvery) {
        p?.start('neighborListBuild')
        this.rebuildNeighborList()
        if (p) await p.end('neighborListBuild', waitForGPU)
      }
    }

    // 3. Compute forces
    p?.start('forceCalculation')
    this.pairStyle.compute(this.state, this.neighborList)
    if (p) await p.end('forceCalculation', waitForGPU)

    // 4. Final integration (half-step velocity)
    p?.start('integration')
    this.integrator.integrateFinal(this.state)
    if (p) await p.end('integration', waitForGPU)

    this.currentStep++
    p?.endStep()
  }

  /**
   * Enable profiling for this simulation
   */
  enableProfiling(): SimulationProfiler {
    this.profiler = new SimulationProfiler()
    return this.profiler
  }

  /**
   * Disable profiling
   */
  disableProfiling(): void {
    this.profiler = null
  }

  /**
   * Read forces back from GPU for debugging
   */
  async readForces(): Promise<Float32Array> {
    return this.state.readForces()
  }

  /**
   * Read velocities back from GPU for debugging
   */
  async readVelocities(): Promise<Float32Array> {
    return this.state.readVelocities()
  }

  /**
   * Get neighbor list stats for debugging
   */
  async getNeighborListStats() {
    const counts = await this.neighborList.readNeighborCounts()
    let totalNeighbors = 0
    let maxNeighborsFound = 0
    for (let i = 0; i < counts.length; i++) {
      totalNeighbors += counts[i]
      maxNeighborsFound = Math.max(maxNeighborsFound, counts[i])
    }

    return {
      maxNeighbors: this.neighborList.maxNeighbors,
      numCells: this.neighborList.cellList.numCells,
      numCellsX: this.neighborList.cellList.numCellsX,
      numCellsY: this.neighborList.cellList.numCellsY,
      numCellsZ: this.neighborList.cellList.numCellsZ,
      avgNeighbors: totalNeighbors / this.numAtoms,
      maxNeighborsFound,
    }
  }

  /**
   * Check if the simulation is stable (no NaNs in positions)
   */
  async checkStability(): Promise<{ stable: boolean; message?: string }> {
    const pos = await this.readPositions()
    for (let i = 0; i < pos.length; i++) {
      if (isNaN(pos[i])) {
        return { stable: false, message: `NaN detected at index ${i}` }
      }
    }
    return { stable: true }
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
   * Compute total energy (kinetic + potential) - ALL ON GPU
   * This requires GPU readback and is async
   */
  async computeEnergy(): Promise<{
    kinetic: number
    potential: number
    total: number
    temperature: number
  }> {
    // Compute kinetic energy on GPU
    const { kineticEnergy: kinetic, temperature } = await this.kineticEnergy.compute(this.state)

    // Compute potential energy on GPU (forces are also computed as a side effect)
    const potential = await this.pairStyle.computeWithEnergy(this.state, this.neighborList)

    return {
      kinetic,
      potential,
      total: kinetic + potential,
      temperature,
    }
  }

  /**
   * Compute kinetic energy and temperature only (faster, no force recalculation)
   * ALL ON GPU
   */
  async computeKineticEnergy(): Promise<{ kinetic: number; temperature: number }> {
    const { kineticEnergy: kinetic, temperature } = await this.kineticEnergy.compute(this.state)
    return { kinetic, temperature }
  }

  /**
   * Destroy GPU resources
   */
  destroy(): void {
    this.state.destroy()
    this.neighborList.destroy()
    this.integrator.destroy()
    this.pairStyle.destroy()
    this.kineticEnergy.destroy()
    this.ctx.destroy()
  }
}

