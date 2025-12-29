import './style.css'
import { Simulation, SimulationVisualizer } from './core'

// UI elements
let simulation: Simulation | null = null
let visualizer: SimulationVisualizer | null = null
let statusEl: HTMLElement | null = null
let playBtn: HTMLButtonElement | null = null
let stepBtn: HTMLButtonElement | null = null
let resetBtn: HTMLButtonElement | null = null
let stepsSlider: HTMLInputElement | null = null
let stepsValue: HTMLElement | null = null

async function initializeApp() {
  const app = document.getElementById('app')!
  
  // Create layout
  app.innerHTML = `
    <div class="app-container">
      <header class="header">
        <h1>WebGPU Molecular Dynamics</h1>
        <p class="subtitle">Lennard-Jones System Simulation</p>
      </header>
      
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
            <h2>Speed</h2>
            <div class="slider-group">
              <label for="steps-slider">Steps/frame:</label>
              <input type="range" id="steps-slider" min="1" max="100" value="10">
              <span id="steps-value">10</span>
            </div>
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
        </aside>
      </div>
    </div>
  `

  // Get UI elements
  statusEl = document.getElementById('status')
  playBtn = document.getElementById('play-btn') as HTMLButtonElement
  stepBtn = document.getElementById('step-btn') as HTMLButtonElement
  resetBtn = document.getElementById('reset-btn') as HTMLButtonElement
  stepsSlider = document.getElementById('steps-slider') as HTMLInputElement
  stepsValue = document.getElementById('steps-value')

  // Set up event listeners
  playBtn?.addEventListener('click', togglePlay)
  stepBtn?.addEventListener('click', singleStep)
  resetBtn?.addEventListener('click', resetSimulation)
  stepsSlider?.addEventListener('input', updateStepsPerFrame)

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
  
  // Clean up existing simulation
  if (visualizer) {
    visualizer.destroy()
  }
  if (simulation) {
    simulation.destroy()
  }

  try {
    // Create a simple LJ liquid
    // 8x8x8 = 512 atoms
    simulation = await Simulation.createLJLiquid(8, 8, 8, {
      density: 0.8,
      temperature: 1.0,
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
  } else {
    visualizer.start({
      stepsPerFrame: parseInt(stepsSlider?.value ?? '10'),
      onStep: (step) => {
        const stepEl = document.getElementById('info-step')
        if (stepEl) stepEl.textContent = step.toString()
      }
    })
    if (playBtn) playBtn.textContent = '⏸ Pause'
    if (stepBtn) stepBtn.disabled = true
  }
}

async function singleStep() {
  if (!visualizer) return
  await visualizer.stepAndRender()
  
  const stepEl = document.getElementById('info-step')
  if (stepEl && simulation) {
    stepEl.textContent = simulation.timestep.toString()
  }
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
