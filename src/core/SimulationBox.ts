import * as THREE from 'three'

/**
 * Represents a simulation box/cell, supporting both orthogonal and triclinic boxes.
 * Compatible with omovi's SimulationCell for visualization.
 */
export class SimulationBox {
  /** Box vector a (x-direction) */
  a: THREE.Vector3
  /** Box vector b (y-direction, may have xy tilt) */
  b: THREE.Vector3
  /** Box vector c (z-direction, may have xz, yz tilts) */
  c: THREE.Vector3
  /** Origin of the box */
  origin: THREE.Vector3
  /** Periodic boundary conditions [x, y, z] */
  periodic: [boolean, boolean, boolean]

  constructor(
    a: THREE.Vector3 = new THREE.Vector3(10, 0, 0),
    b: THREE.Vector3 = new THREE.Vector3(0, 10, 0),
    c: THREE.Vector3 = new THREE.Vector3(0, 0, 10),
    origin: THREE.Vector3 = new THREE.Vector3(0, 0, 0),
    periodic: [boolean, boolean, boolean] = [true, true, true]
  ) {
    this.a = a.clone()
    this.b = b.clone()
    this.c = c.clone()
    this.origin = origin.clone()
    this.periodic = periodic
  }

  /** Create an orthogonal box from dimensions */
  static fromDimensions(
    lx: number,
    ly: number,
    lz: number,
    xlo = 0,
    ylo = 0,
    zlo = 0,
    periodic: [boolean, boolean, boolean] = [true, true, true]
  ): SimulationBox {
    return new SimulationBox(
      new THREE.Vector3(lx, 0, 0),
      new THREE.Vector3(0, ly, 0),
      new THREE.Vector3(0, 0, lz),
      new THREE.Vector3(xlo, ylo, zlo),
      periodic
    )
  }

  /** Create from LAMMPS-style bounds */
  static fromBounds(
    xlo: number,
    xhi: number,
    ylo: number,
    yhi: number,
    zlo: number,
    zhi: number,
    xy = 0,
    xz = 0,
    yz = 0,
    periodic: [boolean, boolean, boolean] = [true, true, true]
  ): SimulationBox {
    return new SimulationBox(
      new THREE.Vector3(xhi - xlo, 0, 0),
      new THREE.Vector3(xy, yhi - ylo, 0),
      new THREE.Vector3(xz, yz, zhi - zlo),
      new THREE.Vector3(xlo, ylo, zlo),
      periodic
    )
  }

  /** Get box dimensions [lx, ly, lz] */
  get dimensions(): [number, number, number] {
    return [this.a.x, this.b.y, this.c.z]
  }

  /** Get box bounds [xlo, xhi, ylo, yhi, zlo, zhi] */
  get bounds(): [number, number, number, number, number, number] {
    const [lx, ly, lz] = this.dimensions
    return [
      this.origin.x,
      this.origin.x + lx,
      this.origin.y,
      this.origin.y + ly,
      this.origin.z,
      this.origin.z + lz,
    ]
  }

  /** Check if box is triclinic (has tilts) */
  get isTriclinic(): boolean {
    return this.b.x !== 0 || this.c.x !== 0 || this.c.y !== 0
  }

  /** Get tilt factors [xy, xz, yz] */
  get tilts(): [number, number, number] {
    return [this.b.x, this.c.x, this.c.y]
  }

  /** Get volume of the box */
  get volume(): number {
    return this.a.dot(this.b.clone().cross(this.c))
  }

  /** Get center of the box */
  getCenter(): THREE.Vector3 {
    return this.origin
      .clone()
      .add(this.a.clone().multiplyScalar(0.5))
      .add(this.b.clone().multiplyScalar(0.5))
      .add(this.c.clone().multiplyScalar(0.5))
  }

  /** Create a GPU-friendly uniform buffer data */
  toGPUData(): Float32Array {
    // Pack box data for GPU: origin (4), a (4), b (4), c (4), dimensions (4)
    // Using vec4 alignment for WebGPU
    return new Float32Array([
      this.origin.x, this.origin.y, this.origin.z, 0, // origin + padding
      this.a.x, this.a.y, this.a.z, 0, // a + padding
      this.b.x, this.b.y, this.b.z, 0, // b + padding
      this.c.x, this.c.y, this.c.z, 0, // c + padding
      ...this.dimensions, 0, // dimensions + padding
      this.periodic[0] ? 1 : 0, this.periodic[1] ? 1 : 0, this.periodic[2] ? 1 : 0, 0, // periodic + padding
    ])
  }

  /** Clone this box */
  clone(): SimulationBox {
    return new SimulationBox(
      this.a.clone(),
      this.b.clone(),
      this.c.clone(),
      this.origin.clone(),
      [...this.periodic]
    )
  }
}

