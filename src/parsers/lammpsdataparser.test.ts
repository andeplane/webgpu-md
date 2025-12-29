import { describe, it, expect } from 'vitest'
import { parseLAMMPSData, generateLAMMPSData } from './lammpsdataparser'
import { SimulationBox } from '../core/SimulationBox'

describe('parseLAMMPSData', () => {
  describe('parsing atomic style', () => {
    it('should parse simple atomic data file', () => {
      // Arrange
      const data = `LAMMPS data file

4 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms # atomic

1 1 1.0 2.0 3.0
2 1 4.0 5.0 6.0
3 1 7.0 8.0 9.0
4 1 0.5 0.5 0.5
`

      // Act
      const result = parseLAMMPSData(data)

      // Assert
      expect(result.numAtoms).toBe(4)
      expect(result.numTypes).toBe(1)
      expect(result.positions.length).toBe(12) // 4 atoms * 3 coords
      expect(result.types.length).toBe(4)
      
      // Check first atom position (sorted by id)
      expect(result.positions[0]).toBeCloseTo(1.0) // x
      expect(result.positions[1]).toBeCloseTo(2.0) // y
      expect(result.positions[2]).toBeCloseTo(3.0) // z
      
      // Types should be 0-indexed
      expect(result.types[0]).toBe(0)
    })

    it('should parse box dimensions correctly', () => {
      // Arrange
      const data = `Test

2 atoms
1 atom types

-5.0 15.0 xlo xhi
0.0 20.0 ylo yhi
-10.0 10.0 zlo zhi

Atoms

1 1 0.0 0.0 0.0
2 1 1.0 1.0 1.0
`

      // Act
      const result = parseLAMMPSData(data)

      // Assert
      const [lx, ly, lz] = result.box.dimensions
      expect(lx).toBeCloseTo(20.0) // 15 - (-5)
      expect(ly).toBeCloseTo(20.0) // 20 - 0
      expect(lz).toBeCloseTo(20.0) // 10 - (-10)
      expect(result.box.origin.x).toBeCloseTo(-5.0)
      expect(result.box.origin.y).toBeCloseTo(0.0)
      expect(result.box.origin.z).toBeCloseTo(-10.0)
    })

    it('should parse multiple atom types', () => {
      // Arrange
      const data = `Multi-type test

3 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms

1 1 0.0 0.0 0.0
2 2 1.0 1.0 1.0
3 1 2.0 2.0 2.0
`

      // Act
      const result = parseLAMMPSData(data)

      // Assert
      expect(result.numTypes).toBe(2)
      // Types are 0-indexed internally
      expect(result.types[0]).toBe(0) // type 1 -> 0
      expect(result.types[1]).toBe(1) // type 2 -> 1
      expect(result.types[2]).toBe(0) // type 1 -> 0
    })
  })

  describe('parsing molecular style', () => {
    it('should parse molecular style with mol-id', () => {
      // Arrange
      const data = `Molecular test

2 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms # molecular

1 1 1 1.0 2.0 3.0
2 1 1 4.0 5.0 6.0
`

      // Act
      const result = parseLAMMPSData(data)

      // Assert
      expect(result.numAtoms).toBe(2)
      expect(result.positions[0]).toBeCloseTo(1.0)
      expect(result.positions[1]).toBeCloseTo(2.0)
      expect(result.positions[2]).toBeCloseTo(3.0)
    })
  })

  describe('parsing masses', () => {
    it('should parse Masses section', () => {
      // Arrange
      const data = `With masses

2 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 12.0
2 16.0

Atoms

1 1 0.0 0.0 0.0
2 2 1.0 1.0 1.0
`

      // Act
      const result = parseLAMMPSData(data)

      // Assert
      expect(result.masses[0]).toBeCloseTo(12.0)
      expect(result.masses[1]).toBeCloseTo(16.0)
    })

    it('should default masses to 1.0 when not specified', () => {
      // Arrange
      const data = `No masses

2 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms

1 1 0.0 0.0 0.0
2 2 1.0 1.0 1.0
`

      // Act
      const result = parseLAMMPSData(data)

      // Assert
      expect(result.masses[0]).toBeCloseTo(1.0)
      expect(result.masses[1]).toBeCloseTo(1.0)
    })
  })

  describe('triclinic boxes', () => {
    it('should parse triclinic box with tilts', () => {
      // Arrange
      const data = `Triclinic

2 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi
1.0 0.5 0.5 xy xz yz

Atoms

1 1 0.0 0.0 0.0
2 1 1.0 1.0 1.0
`

      // Act
      const result = parseLAMMPSData(data)

      // Assert
      const [xy, xz, yz] = result.box.tilts
      expect(xy).toBeCloseTo(1.0)
      expect(xz).toBeCloseTo(0.5)
      expect(yz).toBeCloseTo(0.5)
      expect(result.box.isTriclinic).toBe(true)
    })
  })

  describe('error handling', () => {
    it('should throw error for empty file', () => {
      // Arrange
      const data = `Empty file

0 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi
`

      // Act & Assert
      expect(() => parseLAMMPSData(data)).toThrow('No atoms found')
    })
  })
})

describe('generateLAMMPSData', () => {
  it('should generate valid LAMMPS data file', () => {
    // Arrange
    const box = SimulationBox.fromDimensions(10, 10, 10)
    const positions = new Float32Array([1, 2, 3, 4, 5, 6])
    const types = new Uint32Array([0, 1])
    const masses = new Float32Array([12.0, 16.0])

    // Act
    const output = generateLAMMPSData(2, box, positions, types, masses)

    // Assert
    expect(output).toContain('2 atoms')
    expect(output).toContain('2 atom types')
    expect(output).toContain('0.000000 10.000000 xlo xhi')
    expect(output).toContain('1 12.000000') // mass for type 1
    expect(output).toContain('2 16.000000') // mass for type 2
    expect(output).toContain('Atoms # atomic')
  })

  it('should roundtrip parse and generate', () => {
    // Arrange
    const originalBox = SimulationBox.fromDimensions(15, 20, 25)
    const originalPositions = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    const originalTypes = new Uint32Array([0, 1, 0])

    // Act
    const generated = generateLAMMPSData(3, originalBox, originalPositions, originalTypes)
    const parsed = parseLAMMPSData(generated)

    // Assert
    expect(parsed.numAtoms).toBe(3)
    expect(parsed.box.dimensions[0]).toBeCloseTo(15)
    expect(parsed.box.dimensions[1]).toBeCloseTo(20)
    expect(parsed.box.dimensions[2]).toBeCloseTo(25)
    
    for (let i = 0; i < 9; i++) {
      expect(parsed.positions[i]).toBeCloseTo(originalPositions[i], 5)
    }
  })
})

