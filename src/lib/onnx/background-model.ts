import * as ort from 'onnxruntime-web';

const INPUT_SIZE = 320;
const MEAN = [0.485, 0.456, 0.406] as const;
const STD = [0.229, 0.224, 0.225] as const;
const MODEL_VERSION = '309c8469';
const MODEL_PATH = `models/u2netp-${MODEL_VERSION}.onnx`;
const MODEL_CACHE = `remove-bg-models-${MODEL_VERSION}`;

// U2NetP is a very small salient-object model. These thresholds intentionally
// suppress low-confidence haze while keeping a narrow soft transition around
// confident foreground edges.
const ALPHA_LOW = 0.48;
const ALPHA_HIGH = 0.74;
const SUPPORT_THRESHOLD = 0.58;

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

function smoothstep(edge0: number, edge1: number, value: number) {
  const t = Math.max(0, Math.min(1, (value - edge0) / Math.max(1e-8, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function majorityCleanup(mask: Uint8Array, width: number, height: number) {
  const cleaned = new Uint8Array(mask.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let votes = 0;
      let total = 0;
      for (let oy = -1; oy <= 1; oy++) {
        const yy = y + oy;
        if (yy < 0 || yy >= height) continue;
        for (let ox = -1; ox <= 1; ox++) {
          const xx = x + ox;
          if (xx < 0 || xx >= width) continue;
          total++;
          if (mask[yy * width + xx]) votes++;
        }
      }
      cleaned[y * width + x] = votes >= Math.ceil(total * 0.56) ? 1 : 0;
    }
  }
  return cleaned;
}

function retainMeaningfulComponents(mask: Uint8Array, confidence: Float32Array, width: number, height: number) {
  const visited = new Uint8Array(mask.length);
  const keep = new Uint8Array(mask.length);
  const components: { pixels: number[]; area: number; peak: number; mean: number }[] = [];

  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || visited[start]) continue;
    const stack = [start];
    const pixels: number[] = [];
    let peak = 0;
    let sum = 0;
    visited[start] = 1;

    while (stack.length) {
      const index = stack.pop()!;
      pixels.push(index);
      const c = confidence[index];
      peak = Math.max(peak, c);
      sum += c;
      const x = index % width;
      const y = Math.floor(index / width);
      const neighbors = [index - 1, index + 1, index - width, index + width];
      for (const next of neighbors) {
        if (next < 0 || next >= mask.length || visited[next] || !mask[next]) continue;
        const nx = next % width;
        const ny = Math.floor(next / width);
        if (Math.abs(nx - x) + Math.abs(ny - y) !== 1) continue;
        visited[next] = 1;
        stack.push(next);
      }
    }

    components.push({ pixels, area: pixels.length, peak, mean: sum / pixels.length });
  }

  if (!components.length) return keep;
  components.sort((a, b) => b.area - a.area);
  const imageArea = width * height;
  const largest = components[0].area;

  for (const component of components) {
    const relativeToLargest = component.area / largest;
    const relativeToImage = component.area / imageArea;
    const isStrong = component.peak >= 0.82 && component.mean >= 0.64;
    const isMeaningfulSize = relativeToImage >= 0.0025 || relativeToLargest >= 0.08;
    if ((component === components[0] || isMeaningfulSize) && isStrong) {
      for (const index of component.pixels) keep[index] = 1;
    }
  }

  // Do not accidentally erase the whole result on difficult inputs. In that
  // case fall back to the cleaned support mask and let the soft-alpha stage
  // suppress uncertain background pixels.
  return keep.some((value) => value !== 0) ? keep : mask;
}

function refineMask(raw: Float32Array, width: number, height: number) {
  const pixels = width * height;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < pixels; i++) {
    min = Math.min(min, raw[i]);
    max = Math.max(max, raw[i]);
  }
  const range = Math.max(1e-8, max - min);
  const confidence = new Float32Array(pixels);
  const support = new Uint8Array(pixels);

  for (let i = 0; i < pixels; i++) {
    const normalized = Math.max(0, Math.min(1, (raw[i] - min) / range));
    confidence[i] = normalized;
    support[i] = normalized >= SUPPORT_THRESHOLD ? 1 : 0;
  }

  const cleaned = majorityCleanup(support, width, height);
  const kept = retainMeaningfulComponents(cleaned, confidence, width, height);
  const alpha = new Uint8Array(pixels);

  for (let i = 0; i < pixels; i++) {
    if (!kept[i] && confidence[i] < ALPHA_HIGH) {
      alpha[i] = 0;
      continue;
    }
    const shaped = smoothstep(ALPHA_LOW, ALPHA_HIGH, confidence[i]);
    alpha[i] = Math.round(Math.pow(shaped, 0.82) * 255);
  }

  return alpha;
}

function postprocessMask(tensor: ort.Tensor, width: number, height: number) {
  const dims = tensor.dims.map(Number);
  if (dims.length < 2) throw new Error(`Unexpected U2NetP output shape: [${dims.join(', ')}]`);

  const maskHeight = dims[dims.length - 2];
  const maskWidth = dims[dims.length - 1];
  const pixels = maskWidth * maskHeight;
  const raw = tensorToFloat32(tensor);
  if (raw.length < pixels) throw new Error('U2NetP output tensor is smaller than expected.');

  const refined = refineMask(raw.subarray(0, pixels), maskWidth, maskHeight);
  const source = new OffscreenCanvas(maskWidth, maskHeight);
  const sourceCtx = source.getContext('2d');
  if (!sourceCtx) throw new Error('OffscreenCanvas 2D context is unavailable.');
  const maskImage = sourceCtx.createImageData(maskWidth, maskHeight);

  for (let i = 0; i < pixels; i++) {
    const alpha = refined[i];
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
