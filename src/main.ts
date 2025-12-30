import './style.css'
import { Simulation, SimulationVisualizer } from './core'
import { BenchmarkModal } from './ui/BenchmarkModal'

// UI elements
let simulation: Simulation | null = null
let visualizer: SimulationVisualizer | null = null
let statusEl: HTMLElement | null = null
let playBtn: HTMLButtonElement | null = null
let stepBtn: HTMLButtonElement | null = null
let resetBtn: HTMLButtonElement | null = null
let scalingBenchmarkBtn: HTMLButtonElement | null = null
let showBoxCheckbox: HTMLInputElement | null = null
let stepsSlider: HTMLInputElement | null = null
let stepsValue: HTMLElement | null = null
let initTempInput: HTMLInputElement | null = null
let unitCellsInput: HTMLInputElement | null = null
let densityInput: HTMLInputElement | null = null

// Energy tracking
let initialEnergy: number | null = null
let lastEnergyUpdate = 0
const ENERGY_UPDATE_INTERVAL = 50 // Update energy display every N steps (less frequent for large systems)

async function initializeApp() {
  const app = document.getElementById('app')!
  
  // Create layout
  app.innerHTML = `
    <div class="app-container">
      <header class="header">
        <h1>WebGPU Molecular Dynamics</h1>
        <p class="subtitle">Lennard-Jones System Simulation<br>Vibe coded by Anders Hafreager, University of Oslo</p>
      </header>
      <a href="https://github.com/andeplane/webgpu-md" target="_blank" rel="noopener noreferrer" class="github-corner" aria-label="View source on GitHub">
        <svg viewBox="0 0 250 250" aria-hidden="true"><path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path><path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" class="octo-arm" style="transform-origin: 130px 106px;"></path><path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path></svg>
      </a>
      
      <div class="main-content">
        <div class="viewer-container" id="viewer"></div>
        
        <aside class="sidebar">
          <div class="panel">
            <h2>Controls</h2>
            <div class="button-group">
              <button id="play-btn" disabled>▶ Play</button>
              <button id="step-btn" disabled>Step</button>
              <button id="reset-btn">Reset</button>
            </div>
          </div>
          
          <div class="panel">
            <h2>Initial Conditions</h2>
            <div class="input-group">
              <label for="unit-cells">Unit cells (n³):</label>
              <input type="number" id="unit-cells" value="60" min="2" max="200" step="1">
            </div>
            <div class="input-group">
              <label for="init-temp">Temperature (T*):</label>
              <input type="number" id="init-temp" value="0.8" min="0.001" max="5" step="0.01">
            </div>
            <div class="input-group">
              <label for="density">Density (ρ*):</label>
              <input type="number" id="density" value="0.8" min="0.1" max="2.0" step="0.1">
            </div>
          </div>
          
          <div class="panel">
            <h2>Speed</h2>
            <div class="slider-group">
              <label for="steps-slider">Steps/frame:</label>
              <input type="range" id="steps-slider" min="1" max="100" value="5">
              <span id="steps-value">5</span>
            </div>
          </div>
          
          <div class="panel">
            <h2>Display</h2>
            <label class="checkbox-label">
              <input type="checkbox" id="show-box" checked>
              Show simulation box
            </label>
          </div>
          
          <div class="panel">
            <h2>Benchmark</h2>
            <button id="scaling-benchmark-btn" class="benchmark-btn">Run benchmark</button>
            <div id="benchmark-result" class="benchmark-result"></div>
          </div>
          
          <div class="panel">
            <h2>Status</h2>
            <div id="status" class="status">
              Initializing WebGPU...
            </div>
          </div>
          
          <div class="panel">
            <h2>System</h2>
            <div class="info-grid">
              <span class="label">Atoms:</span>
              <span id="info-atoms">-</span>
              <span class="label">Timestep:</span>
              <span id="info-step">0</span>
              <span class="label">Box size:</span>
              <span id="info-box">-</span>
            </div>
          </div>
          
          <div class="panel">
            <h2>Energy</h2>
            <div class="info-grid energy-grid">
              <span class="label">KE:</span>
              <span id="info-ke">-</span>
              <span class="label">PE:</span>
              <span id="info-pe">-</span>
              <span class="label">Total:</span>
              <span id="info-total" class="highlight">-</span>
              <span class="label">Temp:</span>
              <span id="info-temp">-</span>
            </div>
            <div class="energy-drift">
              <span class="label">ΔE/E₀:</span>
              <span id="info-drift">-</span>
            </div>
          </div>
        </aside>
      </div>
    </div>
  `

  // Get UI elements
  statusEl = document.getElementById('status')
  playBtn = document.getElementById('play-btn') as HTMLButtonElement
  stepBtn = document.getElementById('step-btn') as HTMLButtonElement
  resetBtn = document.getElementById('reset-btn') as HTMLButtonElement
  scalingBenchmarkBtn = document.getElementById('scaling-benchmark-btn') as HTMLButtonElement
  showBoxCheckbox = document.getElementById('show-box') as HTMLInputElement
  stepsSlider = document.getElementById('steps-slider') as HTMLInputElement
  stepsValue = document.getElementById('steps-value')
  initTempInput = document.getElementById('init-temp') as HTMLInputElement
  unitCellsInput = document.getElementById('unit-cells') as HTMLInputElement
  densityInput = document.getElementById('density') as HTMLInputElement

  // Set up event listeners
  playBtn?.addEventListener('click', togglePlay)
  stepBtn?.addEventListener('click', singleStep)
  resetBtn?.addEventListener('click', resetSimulation)
  showBoxCheckbox?.addEventListener('change', toggleShowBox)
  stepsSlider?.addEventListener('input', updateStepsPerFrame)
  
  // Space key to toggle play/pause
  document.addEventListener('keydown', (e) => {
    // Only handle space if not typing in an input field
    if (e.code === 'Space' && e.target instanceof HTMLElement && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
      e.preventDefault()
      if (playBtn && !playBtn.disabled) {
        togglePlay()
      }
    }
  })

  // Create scaling benchmark modal
  const scalingBenchmarkModal = new BenchmarkModal()
  scalingBenchmarkBtn?.addEventListener('click', () => {
    scalingBenchmarkModal.show()
  })

  // Check WebGPU support
  if (!navigator.gpu) {
    showError('WebGPU is not supported in this browser. Please use Chrome 113+ or another WebGPU-enabled browser.')
    return
  }

  // Initialize simulation
  await resetSimulation()
}

async function resetSimulation() {
  updateStatus('Creating LJ liquid system...')
  
  // Stop visualizer if running and reset button state
  if (visualizer) {
    if (visualizer.running) {
      visualizer.stop()
    }
    visualizer.destroy()
  }
  if (simulation) {
    await simulation.destroy()
  }
  
  // Reset button to show Play
  if (playBtn) {
    playBtn.textContent = '▶ Play'
  }
  
  // Reset energy tracking
  initialEnergy = null
  lastEnergyUpdate = 0

  try {
    // Get values from UI inputs
    const initTemp = parseFloat(initTempInput?.value ?? '0.8')
    const unitCells = parseInt(unitCellsInput?.value ?? '60')
    const density = parseFloat(densityInput?.value ?? '0.8')
    
    // Create a simple LJ liquid using FCC lattice
    // n³ FCC unit cells = 4*n³ atoms
    const numAtoms = 4 * unitCells * unitCells * unitCells
    updateStatus(`Creating LJ liquid (${numAtoms} atoms)...`)
    
    simulation = await Simulation.createLJLiquid(unitCells, unitCells, unitCells, {
      density: density,
      temperature: initTemp,
      epsilon: 1.0,
      sigma: 1.0,
      dt: 0.005,
      cutoff: 2.5,
    })

    updateStatus('Setting up visualization...')

    // Create visualizer
    const viewerContainer = document.getElementById('viewer')!
    visualizer = new SimulationVisualizer(simulation, viewerContainer)
    await visualizer.initialize()

    // Update info display
    updateInfo()
    
    // Compute initial energy
    await updateEnergy()

    // Enable controls
    if (playBtn) playBtn.disabled = false
    if (stepBtn) stepBtn.disabled = false

    updateStatus('Ready! Click Play to start simulation.')

  } catch (error) {
    showError(`Failed to initialize: ${error}`)
  }
}

function togglePlay() {
  if (!visualizer) return

  if (visualizer.running) {
    visualizer.stop()
    if (playBtn) playBtn.textContent = '▶ Play'
    if (stepBtn) stepBtn.disabled = false
    // Update energy when stopping
    updateEnergy()
  } else {
    visualizer.start({
      stepsPerFrame: parseInt(stepsSlider?.value ?? '5'),
      onStep: async (step) => {
        const stepEl = document.getElementById('info-step')
        if (stepEl) stepEl.textContent = step.toString()
        
        // Update energy periodically
        if (step - lastEnergyUpdate >= ENERGY_UPDATE_INTERVAL) {
          await updateEnergy()
        }
      }
    })
    if (playBtn) playBtn.textContent = '⏸ Pause'
    if (stepBtn) stepBtn.disabled = true
  }
}

async function singleStep() {
  updateStatus('Stepping...')
  
  if (!visualizer || !simulation) {
    showError('No visualizer or simulation')
    return
  }
  
  try {
    await simulation.step()
    
    // Update visualization
    await visualizer.update()
    visualizer.render()
    
    const stepEl = document.getElementById('info-step')
    if (stepEl) {
      stepEl.textContent = simulation.timestep.toString()
    }
    
    // Update energy display
    await updateEnergy()
    
    updateStatus(`Step ${simulation.timestep} complete`)
  } catch (error) {
    console.error('Error during step:', error)
    showError(`Simulation error: ${error}`)
  }
}

function toggleShowBox() {
  if (!visualizer || !showBoxCheckbox) return
  visualizer.setShowBox(showBoxCheckbox.checked)
}

function updateStepsPerFrame() {
  if (!stepsSlider || !stepsValue || !visualizer) return
  
  const value = parseInt(stepsSlider.value)
  stepsValue.textContent = value.toString()
  visualizer.setStepsPerFrame(value)
}

function updateInfo() {
  if (!simulation) return
  
  const atomsEl = document.getElementById('info-atoms')
  const boxEl = document.getElementById('info-box')
  const stepEl = document.getElementById('info-step')
  
  if (atomsEl) atomsEl.textContent = simulation.numAtoms.toString()
  if (stepEl) stepEl.textContent = simulation.timestep.toString()
  
  if (boxEl) {
    const [lx] = simulation.box.dimensions
    boxEl.textContent = `${lx.toFixed(1)}³`
  }
}

async function updateEnergy() {
  if (!simulation) return
  
  try {
    const energy = await simulation.computeEnergy()
    
    const keEl = document.getElementById('info-ke')
    const peEl = document.getElementById('info-pe')
    const totalEl = document.getElementById('info-total')
    const tempEl = document.getElementById('info-temp')
    const driftEl = document.getElementById('info-drift')
    
    if (keEl) keEl.textContent = energy.kinetic.toFixed(2)
    if (peEl) peEl.textContent = energy.potential.toFixed(2)
    if (totalEl) totalEl.textContent = energy.total.toFixed(2)
    if (tempEl) tempEl.textContent = energy.temperature.toFixed(3)
    
    // Track energy drift
    if (initialEnergy === null) {
      initialEnergy = energy.total
    }
    
    if (driftEl && initialEnergy !== null && Math.abs(initialEnergy) > 0.001) {
      const drift = (energy.total - initialEnergy) / Math.abs(initialEnergy)
      const driftPercent = (drift * 100).toFixed(4)
      const driftClass = Math.abs(drift) < 0.01 ? 'good' : Math.abs(drift) < 0.1 ? 'warn' : 'bad'
      driftEl.textContent = `${driftPercent}%`
      driftEl.className = driftClass
    }
    
    lastEnergyUpdate = simulation.timestep
  } catch (error) {
    console.error('Error computing energy:', error)
  }
}

function updateStatus(message: string) {
  if (statusEl) {
    statusEl.textContent = message
    statusEl.className = 'status'
  }
}

function showError(message: string) {
  if (statusEl) {
    statusEl.textContent = message
    statusEl.className = 'status error'
  }
  if (playBtn) playBtn.disabled = true
  if (stepBtn) stepBtn.disabled = true
}

// Start the app
initializeApp()
