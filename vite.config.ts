import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  base: '/webgpu-md/',
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
})

