import * as ort from 'onnxruntime-web';

const INPUT_SIZE = 320;
const MEAN = [0.485, 0.456, 0.406] as const;
const STD = [0.229, 0.224, 0.225] as const;
const MODEL_VERSION = '309c8469';
const MODEL_PATH = `models/u2netp-${MODEL_VERSION}.onnx`;
const MODEL_CACHE = `remove-bg-models-${MODEL_VERSION}`;

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

async function loadModelBytes() {
  const modelUrl = appUrl(MODEL_PATH);

  if ('caches' in globalThis) {
    try {
      const cache = await caches.open(MODEL_CACHE);
      const cached = await cache.match(modelUrl);
      if (cached) return await cached.arrayBuffer();

      const response = await fetch(modelUrl, { cache: 'force-cache' });
      if (!response.ok) throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
      await cache.put(modelUrl, response.clone());
      return await response.arrayBuffer();
    } catch (error) {
      console.warn('Model Cache Storage failed, using normal browser fetch.', error);
    }
  }

  const response = await fetch(modelUrl, { cache: 'force-cache' });
  if (!response.ok) throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
  return await response.arrayBuffer();
}

function preprocess(image: ImageBitmap) {
  const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('OffscreenCanvas 2D context is unavailable.');

  ctx.drawImage(image, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const rgba = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const plane = INPUT_SIZE * INPUT_SIZE;
  const values = new Float32Array(plane * 3);

  for (let i = 0; i < plane; i++) {
    values[i] = (rgba[i * 4] / 255 - MEAN[0]) / STD[0];
    values[plane + i] = (rgba[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    values[plane * 2 + i] = (rgba[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }

  return new ort.Tensor('float32', values, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function tensorToFloat32(tensor: ort.Tensor) {
  if (tensor.data instanceof Float32Array) return tensor.data;
  return Float32Array.from(tensor.data as ArrayLike<number>);
}

function postprocessMask(tensor: ort.Tensor, width: number, height: number) {
  const dims = tensor.dims.map(Number);
  if (dims.length < 2) throw new Error(`Unexpected U2NetP output shape: [${dims.join(', ')}]`);

  const maskHeight = dims[dims.length - 2];
  const maskWidth = dims[dims.length - 1];
  const pixels = maskWidth * maskHeight;
  const raw = tensorToFloat32(tensor);
  if (raw.length < pixels) throw new Error('U2NetP output tensor is smaller than expected.');

  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < pixels; i++) {
    const value = raw[i];
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const range = Math.max(1e-8, max - min);

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
  private backend: 'wasm' = 'wasm';

  get activeBackend() {
    return this.backend;
  }

  async load() {
    configureWasmRuntime();
    const modelBytes = await loadModelBytes();

    this.session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    this.backend = 'wasm';
  }

  async removeBackground(image: ImageBitmap) {
    if (!this.session) throw new Error('AI model is not loaded.');

    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    if (!inputName || !outputName) throw new Error('U2NetP model has an invalid input/output contract.');

    const input = preprocess(image);
    const outputs = await this.session.run({ [inputName]: input });
    const output = outputs[outputName];
    if (!output) throw new Error(`U2NetP output '${outputName}' was not returned.`);

    return postprocessMask(output, image.width, image.height);
  }
}
