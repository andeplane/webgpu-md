import { Visualizer, Particles } from 'omovi'
import * as THREE from 'three'
import type { Simulation } from './Simulation'
import { createBoxGeometry, disposeBoxGeometry } from '../utils/boxGeometry'

// Re-export omovi types for convenience
export { Visualizer, Particles } from 'omovi'

/**
 * Default colors for atom types (RGB 0-255)
 */
const DEFAULT_TYPE_COLORS = [
  { r: 74, g: 144, b: 217 },   // Blue (type 0)
  { r: 231, g: 76, b: 60 },    // Red (type 1)
  { r: 46, g: 204, b: 113 },   // Green (type 2)
  { r: 155, g: 89, b: 182 },   // Purple (type 3)
  { r: 241, g: 196, b: 15 },   // Yellow (type 4)
  { r: 26, g: 188, b: 156 },   // Teal (type 5)
]

/**
 * Bridges webgpu-md Simulation with omovi Visualizer for real-time rendering
 */
export class SimulationVisualizer {
  readonly simulation: Simulation
  readonly visualizer: Visualizer
  private particles: InstanceType<typeof Particles> | null = null
  private boxGroup: THREE.Group | null = null
  private animationFrameId: number | null = null
  private isRunning = false
  private stepsPerFrame = 10
  private onStep?: (step: number) => void
  private showBox = true

  constructor(simulation: Simulation, container: HTMLElement) {
    this.simulation = simulation
    
    // Create omovi visualizer with the container element
    this.visualizer = new Visualizer({
      domElement: container,
      initialColors: DEFAULT_TYPE_COLORS,
    })
  }

  /**
   * Initialize particles for rendering
   */
  async initialize(): Promise<void> {
    // Create particles object
    this.particles = new Particles(this.simulation.numAtoms)
    
    // Read initial data from GPU
    const positions = await this.simulation.readPositions()
    const types = await this.simulation.readTypes()

    // Copy data to omovi particles
    for (let i = 0; i < this.simulation.numAtoms; i++) {
      this.particles.add(
        positions[i * 3 + 0],
        positions[i * 3 + 1],
        positions[i * 3 + 2],
        i,  // id
        types[i]  // type
      )
    }

    // Add particles to visualizer
    this.visualizer.add(this.particles)

    // Set up atom type radii
    this.setupAtomTypes()

    // Add simulation box wireframe
    this.updateBoxGeometry()

    // Position camera to view system
    this.focusCamera()
  }

  /**
   * Create or update the simulation box wireframe
   */
  private updateBoxGeometry(): void {
    // Remove existing box
    if (this.boxGroup) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(this.visualizer.scene as any).remove(this.boxGroup)
      disposeBoxGeometry(this.boxGroup)
      this.boxGroup = null
    }

    // Create new box if enabled
    if (this.showBox) {
      this.boxGroup = createBoxGeometry(this.simulation.box)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(this.visualizer.scene as any).add(this.boxGroup)
    }
  }

  /**
   * Show or hide the simulation box
   */
  setShowBox(show: boolean): void {
    this.showBox = show
    this.updateBoxGeometry()
    this.visualizer.forceRender = true
  }

  /**
   * Get whether box is shown
   */
  getShowBox(): boolean {
    return this.showBox
  }

  /**
   * Set up default atom type radii
   */
  private setupAtomTypes(): void {
    // Set default radii for atom types
    for (let i = 0; i < DEFAULT_TYPE_COLORS.length; i++) {
      this.visualizer.setRadius(i, 0.5)
    }
  }

  /**
   * Position camera to view the simulation box
   * Note: omovi will auto-focus when particles are added
   */
  private focusCamera(): void {
    // omovi auto-calculates bounds and positions camera
    // We just trigger it by marking particles as updated
    if (this.particles) {
      this.particles.markNeedsUpdate()
      this.visualizer.forceRender = true
    }
    
    // Ensure point light is in scene (may have been removed by previous code)
    if (this.visualizer.pointLight && !this.visualizer.scene.children.includes(this.visualizer.pointLight)) {
      this.visualizer.scene.add(this.visualizer.pointLight)
    }
  }

  /**
   * Update visualization with current simulation state
   */
  async update(): Promise<void> {
    if (!this.particles) return

    // Read positions from GPU
    const positions = await this.simulation.readPositions()

    // Update omovi particles
    this.particles.positions.set(positions)
    this.particles.markNeedsUpdate()
  }

  /**
   * Start the simulation loop with visualization
   */
  start(options: {
    stepsPerFrame?: number
    onStep?: (step: number) => void
  } = {}): void {
    if (this.isRunning) return

    this.stepsPerFrame = options.stepsPerFrame ?? 10
    this.onStep = options.onStep
    this.isRunning = true
    this.loop()
  }

  /**
   * Stop the simulation loop
   */
  stop(): void {
    this.isRunning = false
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId)
      this.animationFrameId = null
    }
  }

  /**
   * Main animation loop
   */
  private loop = async (): Promise<void> => {
    if (!this.isRunning) return

    // Run simulation steps
    for (let i = 0; i < this.stepsPerFrame; i++) {
      this.simulation.step()
    }

    // Update visualization
    await this.update()

    // Notify callback
    if (this.onStep) {
      this.onStep(this.simulation.timestep)
    }

    // Force omovi to re-render
    this.visualizer.forceRender = true

    // Request next frame
    this.animationFrameId = requestAnimationFrame(this.loop)
  }

  /**
   * Run a single step and update visualization
   */
  async stepAndRender(): Promise<void> {
    this.simulation.step()
    await this.update()
    this.visualizer.forceRender = true
  }

  /**
   * Force a re-render
   */
  render(): void {
    this.visualizer.forceRender = true
  }

  /**
   * Set the number of simulation steps per animation frame
   */
  setStepsPerFrame(steps: number): void {
    this.stepsPerFrame = steps
  }

  /**
   * Get current steps per frame
   */
  getStepsPerFrame(): number {
    return this.stepsPerFrame
  }

  /**
   * Check if simulation is running
   */
  get running(): boolean {
    return this.isRunning
  }

  /**
   * Destroy resources
   */
  destroy(): void {
    this.stop()
    
    // Clean up box geometry
    if (this.boxGroup) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(this.visualizer.scene as any).remove(this.boxGroup)
      disposeBoxGeometry(this.boxGroup)
      this.boxGroup = null
    }
    
    this.visualizer.dispose()
  }
}

/**
 * Create a SimulationVisualizer from a canvas element
 */
export async function createSimulationVisualizer(
  simulation: Simulation,
  canvas: HTMLCanvasElement
): Promise<SimulationVisualizer> {
  const sv = new SimulationVisualizer(simulation, canvas)
  await sv.initialize()
  return sv
}

