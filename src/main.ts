import './style.css'

// WebGPU MD Engine - Entry Point
// This will be populated as we implement the simulation

async function main() {
  // Check WebGPU support
  if (!navigator.gpu) {
    document.getElementById('app')!.innerHTML = `
      <div style="color: red; padding: 20px;">
        WebGPU is not supported in this browser. 
        Please use Chrome 113+ or another WebGPU-enabled browser.
      </div>
    `
    return
  }

  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) {
    document.getElementById('app')!.innerHTML = `
      <div style="color: red; padding: 20px;">
        Failed to get WebGPU adapter.
      </div>
    `
    return
  }

  const device = await adapter.requestDevice()
  console.log('WebGPU initialized successfully')
  console.log('Device limits:', device.limits)

  document.getElementById('app')!.innerHTML = `
    <div style="padding: 20px;">
      <h1>WebGPU MD Engine</h1>
      <p>WebGPU initialized successfully!</p>
      <div id="canvas-container" style="width: 100%; height: 600px;"></div>
      <div id="controls"></div>
    </div>
  `
}

main().catch(console.error)
