import * as ort from 'onnxruntime-web';

const INPUT_SIZE = 1024;

function appUrl(path: string) {
  const base = new URL(import.meta.env.BASE_URL, globalThis.location.origin);
  return new URL(path.replace(/^\//, ''), base).href;
}

function configureWasmRuntime() {
  ort.env.wasm.wasmPaths = appUrl('ort/');
  ort.env.wasm.proxy = false;
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated === true
    ? Math.min(4, navigator.hardwareConcurrency || 1)
    : 1;
  ort.env.wasm.simd = true;
}

function preprocess(image: ImageBitmap) {
  const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('OffscreenCanvas 2D context is unavailable.');

  ctx.drawImage(image, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const rgba = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const plane = INPUT_SIZE * INPUT_SIZE;
  const values = new Float32Array(plane * 3);

  // RMBG 1.4 preprocessing from the model card:
  // resize to 1024x1024, RGB / 255, then normalize with mean .5 and std 1.
  for (let i = 0; i < plane; i++) {
    values[i] = rgba[i * 4] / 255 - 0.5;
    values[plane + i] = rgba[i * 4 + 1] / 255 - 0.5;
    values[plane * 2 + i] = rgba[i * 4 + 2] / 255 - 0.5;
  }

  return new ort.Tensor('float32', values, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function tensorToFloat32(tensor: ort.Tensor) {
  if (tensor.data instanceof Float32Array) return tensor.data;
  return Float32Array.from(tensor.data as ArrayLike<number>);
}

function postprocessMask(tensor: ort.Tensor, width: number, height: number) {
  const dims = tensor.dims.map(Number);
  if (dims.length < 2) throw new Error(`Unexpected RMBG output shape: [${dims.join(', ')}]`);

  const maskHeight = dims[dims.length - 2];
  const maskWidth = dims[dims.length - 1];
  const pixels = maskWidth * maskHeight;
  const raw = tensorToFloat32(tensor);
  if (raw.length < pixels) throw new Error('RMBG output tensor is smaller than expected.');

  // Match BRIA's reference postprocessing: min/max normalize the prediction.
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < pixels; i++) {
    const value = raw[i];
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const range = Math.max(1e-6, max - min);

  const source = new OffscreenCanvas(maskWidth, maskHeight);
  const sourceCtx = source.getContext('2d');
  if (!sourceCtx) throw new Error('OffscreenCanvas 2D context is unavailable.');
  const maskImage = sourceCtx.createImageData(maskWidth, maskHeight);
  for (let i = 0; i < pixels; i++) {
    const alpha = Math.max(0, Math.min(255, Math.round(((raw[i] - min) / range) * 255)));
    maskImage.data[i * 4] = alpha;
    maskImage.data[i * 4 + 1] = alpha;
    maskImage.data[i * 4 + 2] = alpha;
    maskImage.data[i * 4 + 3] = 255;
  }
  sourceCtx.putImageData(maskImage, 0, 0);

  const target = new OffscreenCanvas(width, height);
  const targetCtx = target.getContext('2d', { willReadFrequently: true });
  if (!targetCtx) throw new Error('OffscreenCanvas 2D context is unavailable.');
  targetCtx.imageSmoothingEnabled = true;
  targetCtx.imageSmoothingQuality = 'high';
  targetCtx.drawImage(source, 0, 0, width, height);
  const resized = targetCtx.getImageData(0, 0, width, height).data;
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < mask.length; i++) mask[i] = resized[i * 4];
  return mask;
}

export class BackgroundRemovalOnnxModel {
  private session?: ort.InferenceSession;
  private backend: 'webgpu' | 'wasm' = 'wasm';

  get activeBackend() {
    return this.backend;
  }

  async load() {
    configureWasmRuntime();

    const hasWebGpu = typeof navigator !== 'undefined' && 'gpu' in navigator;
    if (hasWebGpu) {
      try {
        this.session = await ort.InferenceSession.create(appUrl('models/rmbg-1.4-fp16.onnx'), {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
        });
        this.backend = 'webgpu';
        return;
      } catch (error) {
        console.warn('RMBG WebGPU initialization failed, falling back to WASM.', error);
      }
    }

    this.session = await ort.InferenceSession.create(appUrl('models/rmbg-1.4-quantized.onnx'), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    this.backend = 'wasm';
  }

  async removeBackground(image: ImageBitmap) {
    if (!this.session) throw new Error('AI model is not loaded.');

    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    if (!inputName || !outputName) throw new Error('RMBG model has an invalid input/output contract.');

    const input = preprocess(image);
    const outputs = await this.session.run({ [inputName]: input });
    const output = outputs[outputName];
    if (!output) throw new Error(`RMBG output '${outputName}' was not returned.`);

    return postprocessMask(output, image.width, image.height);
  }
}
