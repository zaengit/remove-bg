import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const enableCrossOriginIsolation = process.env.VITE_DEV_CROSS_ORIGIN_ISOLATION !== '0';

export default defineConfig({
  plugins: [react()],
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
  server: enableCrossOriginIsolation
    ? {
        headers: {
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'require-corp',
        },
      }
    : undefined,
});
