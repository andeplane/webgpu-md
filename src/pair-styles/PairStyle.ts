import type { WebGPUContext } from '../compute/WebGPUContext'
import type { NeighborList } from '../compute/NeighborList'
import type { SimulationState } from '../core/SimulationState'

/**
 * Pair coefficient for a type pair (i, j)
 */
export interface PairCoeff {
  typeI: number
  typeJ: number
  [key: string]: number  // Additional parameters specific to pair style
}

/**
 * Configuration for a pair style
 */
export interface PairStyleConfig {
  /** Global cutoff distance */
  cutoff: number
  /** Number of atom types */
  numTypes: number
  /** Per-type-pair coefficients */
  coeffs?: PairCoeff[]
}

/**
 * Result of force computation
 */
export interface ForceResult {
  /** Potential energy (if computed) */
  potentialEnergy?: number
  /** Virial tensor components (if computed) */
  virial?: Float32Array
}

/**
 * Abstract base class for pair potentials
 * Implement this to add new force fields (LJ, Morse, Buckingham, etc.)
 */
export abstract class PairStyle {
  protected ctx: WebGPUContext
  protected numAtoms: number
  protected numTypes: number
  protected cutoff: number

  /** Name of the pair style (e.g., "lj/cut") */
  abstract readonly name: string

  /** GPU buffers for pair coefficients */
  protected coeffBuffer: GPUBuffer | null = null

  constructor(ctx: WebGPUContext, numAtoms: number, config: PairStyleConfig) {
    this.ctx = ctx
    this.numAtoms = numAtoms
    this.numTypes = config.numTypes
    this.cutoff = config.cutoff
  }

  /**
   * Get the cutoff distance for this pair style
   */
  getCutoff(): number {
    return this.cutoff
  }

  /**
   * Set pair coefficients for type pair (i, j)
   * Also sets (j, i) for symmetry
   */
  abstract setCoeff(typeI: number, typeJ: number, ...params: number[]): void

  /**
   * Initialize pair coefficients from config
   */
  abstract initCoeffs(coeffs: PairCoeff[]): void

  /**
   * Update for new simulation box
   */
  abstract updateBox(state: SimulationState): void

  /**
   * Compute forces using the neighbor list
   * Forces are accumulated into state.forcesBuffer
   * 
   * @param state - Simulation state with positions, types, forces buffers
   * @param neighborList - Built neighbor list
   * @param computeEnergy - Whether to compute and return potential energy
   * @returns Force computation result (energy, virial if requested)
   */
  abstract compute(
    state: SimulationState,
    neighborList: NeighborList,
    computeEnergy?: boolean
  ): ForceResult

  /**
   * Compute forces and return potential energy (async for GPU readback)
   */
  abstract computeWithEnergy(
    state: SimulationState,
    neighborList: NeighborList
  ): Promise<number>

  /**
   * Get the bind group layout for this pair style's compute shader
   */
  abstract getBindGroupLayout(): GPUBindGroupLayout

  /**
   * Clean up GPU resources
   */
  destroy(): void {
    this.coeffBuffer?.destroy()
  }
}

/**
 * Pair coefficient matrix for type-pair parameters
 * Stores symmetric matrix of coefficients accessed by type pairs
 */
export class PairCoeffMatrix<T extends Record<string, number>> {
  private data: Map<string, T> = new Map()
  readonly numTypes: number

  constructor(numTypes: number) {
    this.numTypes = numTypes
  }

  /**
   * Get key for type pair (symmetric)
   */
  private key(i: number, j: number): string {
    const [a, b] = i <= j ? [i, j] : [j, i]
    return `${a}-${b}`
  }

  /**
   * Set coefficients for type pair
   */
  set(i: number, j: number, coeffs: T): void {
    this.data.set(this.key(i, j), coeffs)
  }

  /**
   * Get coefficients for type pair
   */
  get(i: number, j: number): T | undefined {
    return this.data.get(this.key(i, j))
  }

  /**
   * Check if coefficients are set for type pair
   */
  has(i: number, j: number): boolean {
    return this.data.has(this.key(i, j))
  }

  /**
   * Get all set coefficients
   */
  entries(): [number, number, T][] {
    const result: [number, number, T][] = []
    for (const [key, value] of this.data) {
      const [i, j] = key.split('-').map(Number)
      result.push([i, j, value])
    }
    return result
  }

  /**
   * Pack into a flat array for GPU upload
   * Returns array of size numTypes * numTypes * coeffsPerPair
   * 
   * @param coeffsPerPair - Number of coefficients per type pair
   * @param packer - Function to pack coefficients into array
   */
  packForGPU(coeffsPerPair: number, packer: (coeffs: T) => number[]): Float32Array {
    const data = new Float32Array(this.numTypes * this.numTypes * coeffsPerPair)

    for (const [i, j, coeffs] of this.entries()) {
      const packed = packer(coeffs)
      const idx1 = (i * this.numTypes + j) * coeffsPerPair
      const idx2 = (j * this.numTypes + i) * coeffsPerPair

      for (let k = 0; k < coeffsPerPair; k++) {
        data[idx1 + k] = packed[k]
        data[idx2 + k] = packed[k]  // Symmetric
      }
    }

    return data
  }
}

