import './style.css'
import { Simulation, SimulationVisualizer, runBenchmark } from './core'

// UI elements
let simulation: Simulation | null = null
let visualizer: SimulationVisualizer | null = null
let statusEl: HTMLElement | null = null
let playBtn: HTMLButtonElement | null = null
let stepBtn: HTMLButtonElement | null = null
let resetBtn: HTMLButtonElement | null = null
let benchmarkBtn: HTMLButtonElement | null = null
let showBoxCheckbox: HTMLInputElement | null = null
let stepsSlider: HTMLInputElement | null = null
let stepsValue: HTMLElement | null = null
let initTempInput: HTMLInputElement | null = null
let unitCellsInput: HTMLInputElement | null = null

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
            <h2>Initial Conditions</h2>
            <div class="input-group">
              <label for="init-temp">Temperature (T*):</label>
              <input type="number" id="init-temp" value="0.01" min="0.001" max="5" step="0.01">
            </div>
            <div class="input-group">
              <label for="unit-cells">Unit cells (n³):</label>
              <input type="number" id="unit-cells" value="10" min="2" max="50" step="1">
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
            <h2>Display</h2>
            <label class="checkbox-label">
              <input type="checkbox" id="show-box" checked>
              Show simulation box
            </label>
          </div>
          
          <div class="panel">
            <h2>Benchmark</h2>
            <label class="checkbox-label">
              <input type="checkbox" id="profile-checkbox">
              Enable profiling (slower)
            </label>
            <button id="benchmark-btn" class="benchmark-btn">Run Benchmark</button>
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
  benchmarkBtn = document.getElementById('benchmark-btn') as HTMLButtonElement
  showBoxCheckbox = document.getElementById('show-box') as HTMLInputElement
  stepsSlider = document.getElementById('steps-slider') as HTMLInputElement
  stepsValue = document.getElementById('steps-value')
  initTempInput = document.getElementById('init-temp') as HTMLInputElement
  unitCellsInput = document.getElementById('unit-cells') as HTMLInputElement

  // Set up event listeners
  playBtn?.addEventListener('click', togglePlay)
  stepBtn?.addEventListener('click', singleStep)
  resetBtn?.addEventListener('click', resetSimulation)
  benchmarkBtn?.addEventListener('click', runBenchmarkMode)
  showBoxCheckbox?.addEventListener('change', toggleShowBox)
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
  
  // Reset energy tracking
  initialEnergy = null
  lastEnergyUpdate = 0

  try {
    // Get values from UI inputs
    const initTemp = parseFloat(initTempInput?.value ?? '0.01')
    const unitCells = parseInt(unitCellsInput?.value ?? '10')
    
    // Create a simple LJ liquid using FCC lattice
    // n³ FCC unit cells = 4*n³ atoms
    const numAtoms = 4 * unitCells * unitCells * unitCells
    updateStatus(`Creating LJ liquid (${numAtoms} atoms)...`)
    
    simulation = await Simulation.createLJLiquid(unitCells, unitCells, unitCells, {
      density: 0.8,
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
      stepsPerFrame: parseInt(stepsSlider?.value ?? '10'),
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

async function runBenchmarkMode() {
  if (!benchmarkBtn) return
  
  // Stop visualization if running
  if (visualizer?.running) {
    visualizer.stop()
    if (playBtn) playBtn.textContent = '▶ Play'
    if (stepBtn) stepBtn.disabled = false
  }

  // Disable controls during benchmark
  benchmarkBtn.disabled = true
  if (playBtn) playBtn.disabled = true
  if (stepBtn) stepBtn.disabled = true
  if (resetBtn) resetBtn.disabled = true

  const resultEl = document.getElementById('benchmark-result')
  const profileCheckbox = document.getElementById('profile-checkbox') as HTMLInputElement
  const enableProfiling = profileCheckbox?.checked ?? false
  
  if (resultEl) resultEl.innerHTML = 'Creating simulation...'
  updateStatus('Running benchmark (no visualization)...')

  let benchSim: Simulation | undefined
  try {
    // Create a fresh simulation for benchmark
    // 80x80x80 FCC unit cells = 4*512000 = 2,048,000 atoms
    benchSim = await Simulation.createLJLiquid(80, 80, 80, {
      density: 0.8,
      temperature: 1.0,
      epsilon: 1.0,
      sigma: 1.0,
      dt: 0.005,
      cutoff: 2.5,
    })

    if (resultEl) resultEl.innerHTML = `Running ${benchSim.numAtoms} atoms...`

    const result = await runBenchmark(benchSim, {
      warmupSteps: 100,
      benchmarkSteps: 1000,
      profile: enableProfiling,
    })

    // Display results
    if (resultEl) {
      let html = `
        <div class="bench-stat">
          <span class="bench-label">Atoms:</span>
          <span class="bench-value">${result.numAtoms}</span>
        </div>
        <div class="bench-stat">
          <span class="bench-label">Steps/sec:</span>
          <span class="bench-value">${result.stepsPerSecond.toFixed(1)}</span>
        </div>
        <div class="bench-stat highlight">
          <span class="bench-label">M atom-steps/sec:</span>
          <span class="bench-value">${result.millionAtomStepsPerSecond.toFixed(2)}</span>
        </div>
        <div class="bench-stat">
          <span class="bench-label">ns/atom-step:</span>
          <span class="bench-value">${result.nsPerAtomStep.toFixed(2)}</span>
        </div>
      `
      
      // Add profiling breakdown if available
      if (result.profiling && result.profiling.total > 0) {
        const p = result.profiling
        const pct = (val: number) => ((val / p.total) * 100).toFixed(1)
        const ms = (val: number) => (val / p.stepCount).toFixed(2)
        
        html += `
          <div class="bench-divider"></div>
          <div class="bench-stat">
            <span class="bench-label">Force calc:</span>
            <span class="bench-value">${pct(p.forceCalculation)}% (${ms(p.forceCalculation)} ms)</span>
          </div>
          <div class="bench-stat">
            <span class="bench-label">Neighbor list:</span>
            <span class="bench-value">${pct(p.neighborListBuild)}% (${ms(p.neighborListBuild)} ms)</span>
          </div>
          <div class="bench-stat">
            <span class="bench-label">Integration:</span>
            <span class="bench-value">${pct(p.integration)}% (${ms(p.integration)} ms)</span>
          </div>
        `
      }
      
      resultEl.innerHTML = html
    }

    updateStatus('Benchmark complete!')

  } catch (error) {
    if (resultEl) resultEl.innerHTML = `Error: ${error}`
    updateStatus('Benchmark failed')
  } finally {
    // Clean up benchmark simulation (always runs, even on error)
    benchSim?.destroy()
  }

  // Re-enable controls
  benchmarkBtn.disabled = false
  if (playBtn) playBtn.disabled = false
  if (stepBtn) stepBtn.disabled = false
  if (resetBtn) resetBtn.disabled = false
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
