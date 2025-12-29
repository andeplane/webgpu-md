import { SimulationBox } from '../core/SimulationBox'
import * as THREE from 'three'

/**
 * Parsed atom data from LAMMPS data file
 */
export interface ParsedAtom {
  id: number
  type: number
  x: number
  y: number
  z: number
  molId?: number
}

/**
 * Result of parsing a LAMMPS data file
 */
export interface LAMMPSDataResult {
  /** Number of atoms */
  numAtoms: number
  /** Number of atom types */
  numTypes: number
  /** Simulation box */
  box: SimulationBox
  /** Atom positions (Float32Array, 3 * numAtoms) */
  positions: Float32Array
  /** Atom types (Uint32Array, numAtoms) - 0-indexed */
  types: Uint32Array
  /** Atom IDs (original from file) */
  ids: Uint32Array
  /** Masses per type (Float32Array, numTypes) */
  masses: Float32Array
}

class ParseError extends Error {
  constructor(message: string, lineNumber?: number) {
    super(lineNumber !== undefined ? `Line ${lineNumber}: ${message}` : message)
    this.name = 'ParseError'
  }
}

/**
 * Parse a LAMMPS data file
 * Supports atomic and molecular atom styles
 */
export function parseLAMMPSData(data: string): LAMMPSDataResult {
  const lines = data.split('\n')
  let lineNumber = 0

  // Header parsing state
  let numAtoms = 0
  let numTypes = 0
  let xlo = 0, xhi = 0
  let ylo = 0, yhi = 0
  let zlo = 0, zhi = 0
  let xy = 0, xz = 0, yz = 0

  // Skip title line
  lineNumber = 1

  // Parse header
  while (lineNumber < lines.length) {
    const line = lines[lineNumber].trim()
    lineNumber++

    // Skip empty lines and comments
    if (line === '' || line.startsWith('#')) continue

    // Check for section headers
    if (line === 'Atoms' || line.startsWith('Atoms ')) break
    if (line === 'Masses') {
      // Parse masses section later
      break
    }

    // Parse header values
    if (line.includes('atoms')) {
      numAtoms = parseInt(line.split(/\s+/)[0])
    } else if (line.includes('atom types')) {
      numTypes = parseInt(line.split(/\s+/)[0])
    } else if (line.includes('xlo xhi')) {
      const parts = line.split(/\s+/)
      xlo = parseFloat(parts[0])
      xhi = parseFloat(parts[1])
    } else if (line.includes('ylo yhi')) {
      const parts = line.split(/\s+/)
      ylo = parseFloat(parts[0])
      yhi = parseFloat(parts[1])
    } else if (line.includes('zlo zhi')) {
      const parts = line.split(/\s+/)
      zlo = parseFloat(parts[0])
      zhi = parseFloat(parts[1])
    } else if (line.includes('xy xz yz')) {
      const parts = line.split(/\s+/)
      xy = parseFloat(parts[0])
      xz = parseFloat(parts[1])
      yz = parseFloat(parts[2])
    }
  }

  if (numAtoms === 0) {
    throw new ParseError('No atoms found in file')
  }

  // Initialize masses (default to 1.0)
  const masses = new Float32Array(numTypes)
  masses.fill(1.0)

  // Find and parse Masses section if present
  let massesLineNumber = findSection(lines, 'Masses')
  if (massesLineNumber !== -1) {
    massesLineNumber++
    // Skip empty lines
    while (massesLineNumber < lines.length && lines[massesLineNumber].trim() === '') {
      massesLineNumber++
    }
    // Parse mass entries
    for (let i = 0; i < numTypes && massesLineNumber < lines.length; i++) {
      const line = lines[massesLineNumber].trim()
      if (line === '' || line.startsWith('#')) {
        massesLineNumber++
        i--
        continue
      }
      // Check for next section
      if (!line.match(/^\d/)) break

      const parts = line.split(/\s+/)
      const typeId = parseInt(parts[0]) - 1  // Convert to 0-indexed
      const mass = parseFloat(parts[1])
      if (typeId >= 0 && typeId < numTypes) {
        masses[typeId] = mass
      }
      massesLineNumber++
    }
  }

  // Find Atoms section
  let atomsLineNumber = findSection(lines, 'Atoms')
  if (atomsLineNumber === -1) {
    throw new ParseError('No Atoms section found')
  }

  // Check atom style (atomic or molecular)
  const atomsLine = lines[atomsLineNumber]
  const isMolecular = atomsLine.includes('molecular') || atomsLine.includes('full')

  atomsLineNumber++
  // Skip empty lines after section header
  while (atomsLineNumber < lines.length && lines[atomsLineNumber].trim() === '') {
    atomsLineNumber++
  }

  // Parse atoms
  const positions = new Float32Array(numAtoms * 3)
  const types = new Uint32Array(numAtoms)
  const ids = new Uint32Array(numAtoms)
  const atoms: ParsedAtom[] = []

  for (let i = 0; i < numAtoms && atomsLineNumber < lines.length; i++) {
    const line = lines[atomsLineNumber].trim()
    atomsLineNumber++

    if (line === '' || line.startsWith('#')) {
      i--
      continue
    }

    const parts = line.split(/\s+/).filter(Boolean)
    
    let atom: ParsedAtom
    if (isMolecular) {
      // Format: id mol-id type x y z [ix iy iz]
      atom = {
        id: parseInt(parts[0]),
        molId: parseInt(parts[1]),
        type: parseInt(parts[2]) - 1,  // Convert to 0-indexed
        x: parseFloat(parts[3]),
        y: parseFloat(parts[4]),
        z: parseFloat(parts[5]),
      }
    } else {
      // Format: id type x y z [ix iy iz]
      atom = {
        id: parseInt(parts[0]),
        type: parseInt(parts[1]) - 1,  // Convert to 0-indexed
        x: parseFloat(parts[2]),
        y: parseFloat(parts[3]),
        z: parseFloat(parts[4]),
      }
    }

    atoms.push(atom)
  }

  // Sort by ID and fill arrays
  atoms.sort((a, b) => a.id - b.id)
  
  for (let i = 0; i < atoms.length; i++) {
    const atom = atoms[i]
    positions[i * 3 + 0] = atom.x
    positions[i * 3 + 1] = atom.y
    positions[i * 3 + 2] = atom.z
    types[i] = atom.type
    ids[i] = atom.id
  }

  // Create simulation box
  const box = new SimulationBox(
    new THREE.Vector3(xhi - xlo, 0, 0),
    new THREE.Vector3(xy, yhi - ylo, 0),
    new THREE.Vector3(xz, yz, zhi - zlo),
    new THREE.Vector3(xlo, ylo, zlo)
  )

  return {
    numAtoms,
    numTypes,
    box,
    positions,
    types,
    ids,
    masses,
  }
}

/**
 * Find the line number of a section header
 */
function findSection(lines: string[], sectionName: string): number {
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()
    if (line === sectionName || line.startsWith(sectionName + ' ')) {
      return i
    }
  }
  return -1
}

/**
 * Generate a simple LAMMPS data file string for testing
 */
export function generateLAMMPSData(
  numAtoms: number,
  box: SimulationBox,
  positions: Float32Array,
  types: Uint32Array,
  masses?: Float32Array
): string {
  const [lx, ly, lz] = box.dimensions
  const [xy, xz, yz] = box.tilts
  const numTypes = Math.max(...Array.from(types)) + 1

  let output = 'LAMMPS data file\n\n'
  output += `${numAtoms} atoms\n`
  output += `${numTypes} atom types\n\n`
  
  output += `${box.origin.x.toFixed(6)} ${(box.origin.x + lx).toFixed(6)} xlo xhi\n`
  output += `${box.origin.y.toFixed(6)} ${(box.origin.y + ly).toFixed(6)} ylo yhi\n`
  output += `${box.origin.z.toFixed(6)} ${(box.origin.z + lz).toFixed(6)} zlo zhi\n`
  
  if (xy !== 0 || xz !== 0 || yz !== 0) {
    output += `${xy.toFixed(6)} ${xz.toFixed(6)} ${yz.toFixed(6)} xy xz yz\n`
  }

  // Masses
  if (masses) {
    output += '\nMasses\n\n'
    for (let i = 0; i < numTypes; i++) {
      output += `${i + 1} ${masses[i].toFixed(6)}\n`
    }
  }

  // Atoms
  output += '\nAtoms # atomic\n\n'
  for (let i = 0; i < numAtoms; i++) {
    const x = positions[i * 3 + 0]
    const y = positions[i * 3 + 1]
    const z = positions[i * 3 + 2]
    const type = types[i] + 1  // Convert to 1-indexed
    output += `${i + 1} ${type} ${x.toFixed(6)} ${y.toFixed(6)} ${z.toFixed(6)}\n`
  }

  return output
}

