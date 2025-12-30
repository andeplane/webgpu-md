# WebGPU Molecular Dynamics

A high-performance molecular dynamics simulation engine running entirely in the browser using WebGPU for all computations. Real-time visualization of 1-2 million atoms at interactive framerates.

**Made by Anders Hafreager, University of Oslo**  
*Vibe coded*

## Try it out

**[Live Demo](https://andeplane.github.io/webgpu-md)** - Run molecular dynamics simulations directly in your browser!

## Features

- **Full WebGPU Compute Pipeline**: Forces, neighbor lists, integration, and kinetic energy calculations all run on the GPU
- **Real-time Visualization**: Simulate and visualize 1-2 million atoms at interactive framerates using [omovi](https://github.com/andeplane/omovi)
- **Benchmark Mode**: Built-in scaling benchmark from 4K to 4M atoms with optional profiling breakdown
- **Lennard-Jones Potential**: Extensible pair style system for adding new force fields
- **LAMMPS Data File Support**: Parse and load LAMMPS data format files
- **Interactive Controls**: Play/pause, step, reset with adjustable simulation parameters
- **Configurable Parameters**: Temperature, density, number of atoms (unit cells), and steps per frame
- **SSAO Post-processing**: Screen-space ambient occlusion for enhanced depth perception

## Visualization

This project uses [omovi](https://github.com/andeplane/omovi) (Online Molecular Visualizer) for high-performance 3D rendering. Omovi provides GPU-accelerated particle rendering with interactive camera controls, particle selection, and visual effects.

## Performance

- **Real-time simulation**: 1-2 million atoms with visualization at interactive framerates
- **Scaling benchmark**: Test performance from 4,000 to 4,000,000 atoms
- **Profiling support**: Detailed timing breakdown of force calculation, neighbor list building, and integration steps
- **Zero-copy rendering**: Direct GPU buffer access for efficient visualization

## Requirements

- Modern browser with WebGPU support (Chrome 113+, Edge 113+, or Firefox Nightly with WebGPU enabled)
- Node.js 18+ (for development)

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
├── core/              # Core simulation engine (Simulation, SimulationState, SimulationBox)
│                      # Benchmark and profiling utilities
├── compute/           # WebGPU compute infrastructure (WebGPUContext, NeighborList, CellList)
├── pair-styles/       # Force field implementations (Lennard-Jones, extensible system)
├── integrators/       # Time integration algorithms (Velocity Verlet)
├── parsers/           # File format parsers (LAMMPS data)
├── shaders/           # WGSL compute shaders (forces, neighbor lists, integration)
├── ui/                # UI components (BenchmarkModal)
├── utils/              # Utility functions (box geometry)
└── main.ts            # Entry point with interactive demo
```

## Usage

The application provides an interactive web interface where you can:

1. **Configure initial conditions**: Set temperature, density, and number of unit cells
2. **Control simulation**: Play, pause, step, or reset the simulation
3. **Adjust speed**: Control steps per frame for performance vs. visualization tradeoff
4. **Run benchmarks**: Test scaling performance across different system sizes
5. **Visualize**: Real-time 3D rendering with camera controls and particle selection

## License

MIT

## Credits

**Made by Anders Hafreager, University of Oslo**  
*Vibe coded*
