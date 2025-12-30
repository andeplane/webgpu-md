import { Chart, registerables } from 'chart.js'
import { runScalingBenchmark, type ScalingResult } from '../core/ScalingBenchmark'

// Register Chart.js components
Chart.register(...registerables)

/**
 * Create and manage the scaling benchmark modal
 */
export class BenchmarkModal {
  private modal: HTMLElement | null = null
  private chart: Chart | null = null
  private results: ScalingResult[] = []
  private profileEnabled = false
  private yAxisMetric: 'stepsPerSec' | 'mAtomStepsPerSec' = 'mAtomStepsPerSec'
  private isRunning = false

  /**
   * Show the modal and start the benchmark
   */
  show(): void {
    this.createModal()
    this.setupChart()
    this.modal?.classList.add('active')
  }

  /**
   * Hide the modal
   */
  hide(): void {
    this.modal?.classList.remove('active')
    if (this.chart) {
      this.chart.destroy()
      this.chart = null
    }
    this.results = []
  }

  /**
   * Create the modal HTML structure
   */
  private createModal(): void {
    // Remove existing modal if present
    const existing = document.getElementById('benchmark-modal')
    if (existing) {
      existing.remove()
    }

    const modal = document.createElement('div')
    modal.id = 'benchmark-modal'
    modal.className = 'benchmark-modal'
    modal.innerHTML = `
      <div class="benchmark-modal-backdrop"></div>
      <div class="benchmark-modal-content">
        <div class="benchmark-modal-header">
          <h2>Scaling Benchmark Results</h2>
          <button class="benchmark-modal-close" id="benchmark-modal-close">×</button>
        </div>
        <div class="benchmark-modal-body">
          <div class="benchmark-modal-controls">
            <label class="checkbox-label">
              <input type="checkbox" id="benchmark-profile-checkbox">
              Enable detailed profiling (slower)
            </label>
            <button class="benchmark-modal-start-btn" id="benchmark-modal-start-btn">Start Benchmark</button>
          </div>
          <div class="benchmark-modal-progress" id="benchmark-modal-progress" style="display: none;">
            <div class="progress-text">Preparing...</div>
          </div>
          <div class="benchmark-modal-chart-container">
            <canvas id="benchmark-chart"></canvas>
          </div>
          <div class="benchmark-modal-chart-toggle">
            <label>
              <input type="radio" name="y-axis-metric" value="stepsPerSec">
              Steps/sec
            </label>
            <label>
              <input type="radio" name="y-axis-metric" value="mAtomStepsPerSec" checked>
              M atom-steps/sec
            </label>
          </div>
          <div class="benchmark-modal-table-container">
            <table class="benchmark-table" id="benchmark-table">
              <thead>
                <tr>
                  <th>Unit Cells</th>
                  <th>Atoms</th>
                  <th>Steps/sec</th>
                  <th>M atom-steps/sec</th>
                  <th class="profiling-col" style="display: none;">Force %</th>
                  <th class="profiling-col" style="display: none;">Neigh %</th>
                  <th class="profiling-col" style="display: none;">Int %</th>
                </tr>
              </thead>
              <tbody id="benchmark-table-body">
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `

    document.body.appendChild(modal)
    this.modal = modal

    // Setup event listeners
    const closeBtn = document.getElementById('benchmark-modal-close')
    const backdrop = modal.querySelector('.benchmark-modal-backdrop')
    const startBtn = document.getElementById('benchmark-modal-start-btn')
    const profileCheckbox = document.getElementById('benchmark-profile-checkbox') as HTMLInputElement
    const yAxisRadios = modal.querySelectorAll('input[name="y-axis-metric"]') as NodeListOf<HTMLInputElement>

    closeBtn?.addEventListener('click', () => {
      if (!this.isRunning) {
        this.hide()
      }
    })

    backdrop?.addEventListener('click', () => {
      if (!this.isRunning) {
        this.hide()
      }
    })

    startBtn?.addEventListener('click', () => {
      this.startBenchmark()
    })

    profileCheckbox?.addEventListener('change', (e) => {
      this.profileEnabled = (e.target as HTMLInputElement).checked
      this.updateProfilingColumns()
    })

    yAxisRadios.forEach((radio) => {
      radio.addEventListener('change', (e) => {
        if ((e.target as HTMLInputElement).checked) {
          this.yAxisMetric = (e.target as HTMLInputElement).value as 'stepsPerSec' | 'mAtomStepsPerSec'
          this.updateChart()
        }
      })
    })
  }

  /**
   * Setup the Chart.js chart
   */
  private setupChart(): void {
    const canvas = document.getElementById('benchmark-chart') as HTMLCanvasElement
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: this.yAxisMetric === 'stepsPerSec' ? 'Steps/sec' : 'M atom-steps/sec',
            data: [],
            borderColor: '#00d4ff',
            backgroundColor: 'rgba(0, 212, 255, 0.1)',
            tension: 0.4,
            pointRadius: 4,
            pointHoverRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            backgroundColor: 'rgba(26, 26, 46, 0.95)',
            titleColor: '#f0f0f5',
            bodyColor: '#f0f0f5',
            borderColor: '#00d4ff',
            borderWidth: 1,
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Number of Atoms',
              color: '#a0a0b0',
            },
            ticks: {
              color: '#a0a0b0',
            },
            grid: {
              color: 'rgba(160, 160, 176, 0.1)',
            },
          },
          y: {
            title: {
              display: true,
              text: this.yAxisMetric === 'stepsPerSec' ? 'Steps/sec' : 'M atom-steps/sec',
              color: '#a0a0b0',
            },
            ticks: {
              color: '#a0a0b0',
            },
            grid: {
              color: 'rgba(160, 160, 176, 0.1)',
            },
          },
        },
      },
    })
  }

  /**
   * Start the benchmark
   */
  private async startBenchmark(): Promise<void> {
    if (this.isRunning) return

    const startBtn = document.getElementById('benchmark-modal-start-btn') as HTMLButtonElement
    const progressEl = document.getElementById('benchmark-modal-progress')
    const profileCheckbox = document.getElementById('benchmark-profile-checkbox') as HTMLInputElement

    this.isRunning = true
    this.results = []
    startBtn.disabled = true
    startBtn.textContent = 'Running...'
    if (progressEl) progressEl.style.display = 'block'

    this.profileEnabled = profileCheckbox?.checked ?? false

    try {
      await runScalingBenchmark(
        { profile: this.profileEnabled },
        (result, index, total) => {
          this.results.push(result)
          this.updateTable()
          this.updateChart()
          this.updateProgress(index + 1, total, result.unitCells)
        }
      )
    } catch (error) {
      console.error('Benchmark failed:', error)
      this.updateProgress(-1, 0, 0, `Error: ${error}`)
    } finally {
      this.isRunning = false
      startBtn.disabled = false
      startBtn.textContent = 'Start Benchmark'
      if (progressEl) progressEl.style.display = 'none'
    }
  }

  /**
   * Update the progress indicator
   */
  private updateProgress(current: number, total: number, unitCells: number, error?: string): void {
    const progressEl = document.getElementById('benchmark-modal-progress')
    if (!progressEl) return

    const progressText = progressEl.querySelector('.progress-text') as HTMLElement
    if (!progressText) return

    if (error) {
      progressText.textContent = error
      return
    }

    if (current > 0) {
      progressText.textContent = `Progress: ${current}/${total} - Running ${unitCells}×${unitCells}×${unitCells}...`
    } else {
      progressText.textContent = 'Preparing...'
    }
  }

  /**
   * Update the results table
   */
  private updateTable(): void {
    const tbody = document.getElementById('benchmark-table-body')
    if (!tbody) return

    tbody.innerHTML = this.results
      .map((result) => {
        const profilingCells = this.profileEnabled && result.profiling
          ? `
            <td>${result.profiling.forceCalcPercent.toFixed(1)}%</td>
            <td>${result.profiling.neighborListPercent.toFixed(1)}%</td>
            <td>${result.profiling.integrationPercent.toFixed(1)}%</td>
          `
          : '<td></td><td></td><td></td>'

        return `
          <tr>
            <td>${result.unitCells}×${result.unitCells}×${result.unitCells}</td>
            <td>${result.numAtoms.toLocaleString()}</td>
            <td>${result.stepsPerSec.toFixed(1)}</td>
            <td>${result.mAtomStepsPerSec.toFixed(2)}</td>
            ${profilingCells}
          </tr>
        `
      })
      .join('')
  }

  /**
   * Update the chart with current results
   */
  private updateChart(): void {
    if (!this.chart) return

    const labels = this.results.map((r) => r.numAtoms.toString())
    const data = this.results.map((r) =>
      this.yAxisMetric === 'stepsPerSec' ? r.stepsPerSec : r.mAtomStepsPerSec
    )

    this.chart.data.labels = labels
    this.chart.data.datasets[0].data = data
    this.chart.data.datasets[0].label =
      this.yAxisMetric === 'stepsPerSec' ? 'Steps/sec' : 'M atom-steps/sec'

    const yAxisTitle =
      this.yAxisMetric === 'stepsPerSec' ? 'Steps/sec' : 'M atom-steps/sec'
    
    // Update Y-axis title
    if (this.chart.options.scales && this.chart.options.scales.y) {
      const yScale = this.chart.options.scales.y as any
      if (yScale.title) {
        yScale.title.text = yAxisTitle
      }
    }

    this.chart.update()
  }

  /**
   * Show/hide profiling columns based on checkbox state
   */
  private updateProfilingColumns(): void {
    const cols = document.querySelectorAll('.profiling-col')
    cols.forEach((col) => {
      ;(col as HTMLElement).style.display = this.profileEnabled ? '' : 'none'
    })
  }
}

