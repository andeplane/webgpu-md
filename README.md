# WebGPU Molecular Dynamics

A high-performance molecular dynamics simulation engine using WebGPU for all computations (forces, neighbor lists, integration), with real-time visualization through [omovi](https://github.com/andeplane/omovi).

## Features

- **Full WebGPU Compute**: Forces, neighbor lists, and integration all run on the GPU
- **Lennard-Jones potential**: Extensible pair style system for adding new potentials
- **Real-time visualization**: Direct integration with omovi for zero-copy rendering
- **LAMMPS file support**: Read LAMMPS data files

## Requirements

- Modern browser with WebGPU support (Chrome 113+, Edge 113+, or Firefox Nightly with WebGPU enabled)
- Node.js 18+

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm test

# Build for production
npm run build
```

## Architecture

```
src/
├── core/           # Core data structures (SimulationState, SimulationBox)
├── compute/        # WebGPU compute infrastructure
├── pair-styles/    # Force field implementations (LJ, etc.)
├── integrators/    # Time integration (Velocity Verlet)
├── parsers/        # File format parsers (LAMMPS data)
├── shaders/        # WGSL compute shaders
└── main.ts         # Entry point with demo
```

## License

MIT

