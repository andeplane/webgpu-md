import { SimulationBox } from './SimulationBox'

/**
 * GPU buffer configuration
 */
export interface GPUBufferConfig {
  device: GPUDevice
  numAtoms: number
  maxNeighbors?: number
}

/**
 * Manages GPU buffers for molecular dynamics simulation.
 * Buffer layouts are designed for efficient WebGPU access and omovi compatibility.
 */
export class SimulationState {
  readonly device: GPUDevice
  readonly numAtoms: number
  readonly maxNeighbors: number

  // Core particle data buffers
  /** Positions: Float32Array[numAtoms * 3] - x, y, z interleaved */
  positionsBuffer: GPUBuffer
  /** Velocities: Float32Array[numAtoms * 3] - vx, vy, vz interleaved */
  velocitiesBuffer: GPUBuffer
  /** Forces: Float32Array[numAtoms * 3] - fx, fy, fz interleaved */
  forcesBuffer: GPUBuffer
  /** Atom types: Uint32Array[numAtoms] */
  typesBuffer: GPUBuffer
  /** Atom masses (per-atom): Float32Array[numAtoms] */
  massesBuffer: GPUBuffer

  // Staging buffers for CPU read-back
  positionsStagingBuffer: GPUBuffer
  typesStagingBuffer: GPUBuffer

  // Simulation box
  box: SimulationBox
  boxBuffer: GPUBuffer

  // CPU-side arrays for initialization and readback
  private _positions: Float32Array
  private _velocities: Float32Array
  private _forces: Float32Array
  private _types: Uint32Array
  private _masses: Float32Array

  constructor(config: GPUBufferConfig) {
    this.device = config.device
    this.numAtoms = config.numAtoms
    this.maxNeighbors = config.maxNeighbors ?? 128

    // Initialize CPU arrays
    this._positions = new Float32Array(config.numAtoms * 3)
    this._velocities = new Float32Array(config.numAtoms * 3)
    this._forces = new Float32Array(config.numAtoms * 3)
    this._types = new Uint32Array(config.numAtoms)
    this._masses = new Float32Array(config.numAtoms)

    // Default masses to 1.0
    this._masses.fill(1.0)

    // Create GPU buffers
    this.positionsBuffer = this.createBuffer(
      this._positions.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      'positions'
    )

    this.velocitiesBuffer = this.createBuffer(
      this._velocities.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      'velocities'
    )

    this.forcesBuffer = this.createBuffer(
      this._forces.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      'forces'
    )

    this.typesBuffer = this.createBuffer(
      this._types.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      'types'
    )

    this.massesBuffer = this.createBuffer(
      this._masses.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      'masses'
    )

    // Create staging buffers for read-back
    this.positionsStagingBuffer = this.createBuffer(
      this._positions.byteLength,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      'positions-staging'
    )

    this.typesStagingBuffer = this.createBuffer(
      this._types.byteLength,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      'types-staging'
    )

    // Initialize simulation box
    this.box = new SimulationBox()
    this.boxBuffer = this.createBuffer(
      this.box.toGPUData().byteLength,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      'box'
    )
  }

  private createBuffer(size: number, usage: GPUBufferUsageFlags, label: string): GPUBuffer {
    return this.device.createBuffer({
      size,
      usage,
      label,
    })
  }

  private writeBuffer(buffer: GPUBuffer, data: ArrayBufferView): void {
    this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength)
  }

  /** Set positions from CPU array */
  setPositions(positions: Float32Array): void {
    if (positions.length !== this._positions.length) {
      throw new Error(`Position array length mismatch: expected ${this._positions.length}, got ${positions.length}`)
    }
    this._positions.set(positions)
    this.writeBuffer(this.positionsBuffer, this._positions)
  }

  /** Set velocities from CPU array */
  setVelocities(velocities: Float32Array): void {
    if (velocities.length !== this._velocities.length) {
      throw new Error(`Velocity array length mismatch: expected ${this._velocities.length}, got ${velocities.length}`)
    }
    this._velocities.set(velocities)
    this.writeBuffer(this.velocitiesBuffer, this._velocities)
  }

  /** Set atom types from CPU array */
  setTypes(types: Uint32Array): void {
    if (types.length !== this._types.length) {
      throw new Error(`Types array length mismatch: expected ${this._types.length}, got ${types.length}`)
    }
    this._types.set(types)
    this.writeBuffer(this.typesBuffer, this._types)
  }

  /** Set masses from CPU array */
  setMasses(masses: Float32Array): void {
    if (masses.length !== this._masses.length) {
      throw new Error(`Masses array length mismatch: expected ${this._masses.length}, got ${masses.length}`)
    }
    this._masses.set(masses)
    this.writeBuffer(this.massesBuffer, this._masses)
  }

  /** Update simulation box */
  setBox(box: SimulationBox): void {
    this.box = box.clone()
    const boxData = this.box.toGPUData()
    this.writeBuffer(this.boxBuffer, boxData)
  }

  /** Read positions back from GPU (async) */
  async readPositions(): Promise<Float32Array> {
    // Wait for any pending GPU work to complete
    await this.device.queue.onSubmittedWorkDone()
    
    const commandEncoder = this.device.createCommandEncoder()
    commandEncoder.copyBufferToBuffer(
      this.positionsBuffer,
      0,
      this.positionsStagingBuffer,
      0,
      this._positions.byteLength
    )
    this.device.queue.submit([commandEncoder.finish()])

    await this.positionsStagingBuffer.mapAsync(GPUMapMode.READ)
    const data = new Float32Array(this.positionsStagingBuffer.getMappedRange().slice(0))
    this.positionsStagingBuffer.unmap()

    this._positions.set(data)
    return this._positions
  }

  /** Read types back from GPU (async) */
  async readTypes(): Promise<Uint32Array> {
    const commandEncoder = this.device.createCommandEncoder()
    commandEncoder.copyBufferToBuffer(
      this.typesBuffer,
      0,
      this.typesStagingBuffer,
      0,
      this._types.byteLength
    )
    this.device.queue.submit([commandEncoder.finish()])

    await this.typesStagingBuffer.mapAsync(GPUMapMode.READ)
    const data = new Uint32Array(this.typesStagingBuffer.getMappedRange().slice(0))
    this.typesStagingBuffer.unmap()

    this._types.set(data)
    return this._types
  }

  /** Get CPU positions (may be stale, call readPositions() first) */
  get positions(): Float32Array {
    return this._positions
  }

  /** Get CPU velocities */
  get velocities(): Float32Array {
    return this._velocities
  }

  /** Get CPU types (may be stale, call readTypes() first) */
  get types(): Uint32Array {
    return this._types
  }

  /** Get CPU masses */
  get masses(): Float32Array {
    return this._masses
  }

  /** Zero out the forces buffer */
  zeroForces(): void {
    this._forces.fill(0)
    this.writeBuffer(this.forcesBuffer, this._forces)
  }

  /** Initialize atoms on a simple cubic lattice */
  initializeLattice(
    nx: number,
    ny: number,
    nz: number,
    spacing: number,
    type = 0
  ): void {
    const expectedAtoms = nx * ny * nz
    if (expectedAtoms !== this.numAtoms) {
      throw new Error(`Lattice size ${expectedAtoms} doesn't match numAtoms ${this.numAtoms}`)
    }

    let idx = 0
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const x = (ix + 0.5) * spacing
          const y = (iy + 0.5) * spacing
          const z = (iz + 0.5) * spacing

          this._positions[idx * 3 + 0] = x
          this._positions[idx * 3 + 1] = y
          this._positions[idx * 3 + 2] = z
          this._types[idx] = type
          idx++
        }
      }
    }

    // Update box to fit lattice
    const lx = nx * spacing
    const ly = ny * spacing
    const lz = nz * spacing
    this.box = SimulationBox.fromDimensions(lx, ly, lz)

    // Upload to GPU
    this.writeBuffer(this.positionsBuffer, this._positions)
    this.writeBuffer(this.typesBuffer, this._types)
    this.writeBuffer(this.boxBuffer, this.box.toGPUData())
  }

  /**
   * Initialize atoms on an FCC lattice
   * FCC has 4 atoms per unit cell at positions:
   * (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
   * 
   * @param nx - number of unit cells in x
   * @param ny - number of unit cells in y  
   * @param nz - number of unit cells in z
   * @param latticeConstant - FCC lattice constant (unit cell size)
   * @param type - atom type (default 0)
   */
  initializeFCC(
    nx: number,
    ny: number,
    nz: number,
    latticeConstant: number,
    type = 0
  ): void {
    const expectedAtoms = 4 * nx * ny * nz
    if (expectedAtoms !== this.numAtoms) {
      throw new Error(`FCC lattice ${nx}x${ny}x${nz} has ${expectedAtoms} atoms, but numAtoms is ${this.numAtoms}`)
    }

    // FCC basis vectors (in units of lattice constant)
    const basis = [
      [0.0, 0.0, 0.0],
      [0.5, 0.5, 0.0],
      [0.5, 0.0, 0.5],
      [0.0, 0.5, 0.5],
    ]

    let idx = 0
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          for (const [bx, by, bz] of basis) {
            const x = (ix + bx) * latticeConstant
            const y = (iy + by) * latticeConstant
            const z = (iz + bz) * latticeConstant

            this._positions[idx * 3 + 0] = x
            this._positions[idx * 3 + 1] = y
            this._positions[idx * 3 + 2] = z
            this._types[idx] = type
            idx++
          }
        }
      }
    }

    // Update box to fit lattice
    const lx = nx * latticeConstant
    const ly = ny * latticeConstant
    const lz = nz * latticeConstant
    this.box = SimulationBox.fromDimensions(lx, ly, lz)

    // Upload to GPU
    this.writeBuffer(this.positionsBuffer, this._positions)
    this.writeBuffer(this.typesBuffer, this._types)
    this.writeBuffer(this.boxBuffer, this.box.toGPUData())
  }

  /** Initialize velocities with Maxwell-Boltzmann distribution */
  initializeVelocities(temperature: number, seed = 12345): void {
    // Simple Box-Muller transform for Gaussian random numbers
    const random = this.seededRandom(seed)

    let sumVx = 0, sumVy = 0, sumVz = 0

    for (let i = 0; i < this.numAtoms; i++) {
      const mass = this._masses[i]
      const sigma = Math.sqrt(temperature / mass)

      // Box-Muller transform
      const u1 = random()
      const u2 = random()
      const u3 = random()
      const u4 = random()

      const vx = sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
      const vy = sigma * Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2)
      const vz = sigma * Math.sqrt(-2 * Math.log(u3)) * Math.cos(2 * Math.PI * u4)

      this._velocities[i * 3 + 0] = vx
      this._velocities[i * 3 + 1] = vy
      this._velocities[i * 3 + 2] = vz

      sumVx += vx * mass
      sumVy += vy * mass
      sumVz += vz * mass
    }

    // Remove center of mass motion
    let totalMass = 0
    for (let i = 0; i < this.numAtoms; i++) {
      totalMass += this._masses[i]
    }

    const vcmX = sumVx / totalMass
    const vcmY = sumVy / totalMass
    const vcmZ = sumVz / totalMass

    for (let i = 0; i < this.numAtoms; i++) {
      this._velocities[i * 3 + 0] -= vcmX
      this._velocities[i * 3 + 1] -= vcmY
      this._velocities[i * 3 + 2] -= vcmZ
    }

    // Upload to GPU
    this.writeBuffer(this.velocitiesBuffer, this._velocities)
  }

  private seededRandom(seed: number): () => number {
    return () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff
      return seed / 0x7fffffff
    }
  }

  /** Destroy all GPU resources */
  destroy(): void {
    this.positionsBuffer.destroy()
    this.velocitiesBuffer.destroy()
    this.forcesBuffer.destroy()
    this.typesBuffer.destroy()
    this.massesBuffer.destroy()
    this.positionsStagingBuffer.destroy()
    this.typesStagingBuffer.destroy()
    this.boxBuffer.destroy()
  }
}

