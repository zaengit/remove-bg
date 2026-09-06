import * as ort from 'onnxruntime-web';
import type { BoundingBox, ImageEmbedding, InteractiveSegmentationModel, PromptPoint, SegmentationResult } from '../../types/editor';

const ENCODER_URL = import.meta.env.VITE_ENCODER_MODEL_URL || '/models/mobile_sam_encoder.onnx';
const DECODER_URL = import.meta.env.VITE_DECODER_MODEL_URL || '/models/mobile_sam_decoder.onnx';
const INPUT_SIZE = Number(import.meta.env.VITE_MODEL_INPUT_SIZE || 1024);

async function createSession(url: string): Promise<ort.InferenceSession> {
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
  ort.env.wasm.simd = true;
  try {
    if ('gpu' in navigator) return await ort.InferenceSession.create(url, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' });
  } catch (error) { console.warn('WebGPU ONNX failed; falling back to WASM.', error); }
  return ort.InferenceSession.create(url, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
}

function resizeImage(image: ImageBitmap, size: number) {
  const scale = size / Math.max(image.width, image.height);
  const width = Math.round(image.width * scale);
  const height = Math.round(image.height * scale);
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(image, 0, 0, width, height);
  return { data: ctx.getImageData(0, 0, size, size).data, scale };
}

function toNchw(data: Uint8ClampedArray, size: number) {
  const out = new Float32Array(3 * size * size);
  const mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375];
  const plane = size * size;
  for (let i = 0; i < plane; i++) {
    out[i] = (data[i * 4] - mean[0]) / std[0];
    out[plane + i] = (data[i * 4 + 1] - mean[1]) / std[1];
    out[plane * 2 + i] = (data[i * 4 + 2] - mean[2]) / std[2];
  }
  return out;
}

function maskToOriginal(values: Float32Array, mw: number, mh: number, embedding: ImageEmbedding): Uint8Array {
  const src = new OffscreenCanvas(mw, mh), sctx = src.getContext('2d')!;
  const image = sctx.createImageData(mw, mh);
  for (let i = 0; i < values.length; i++) {
    const v = values[i] > 0 ? 255 : 0;
    image.data[i * 4] = image.data[i * 4 + 1] = image.data[i * 4 + 2] = v; image.data[i * 4 + 3] = 255;
  }
  sctx.putImageData(image, 0, 0);
  const resizedW = Math.round(embedding.originalWidth * embedding.scale);
  const resizedH = Math.round(embedding.originalHeight * embedding.scale);
  const crop = new OffscreenCanvas(resizedW, resizedH);
  crop.getContext('2d')!.drawImage(src, 0, 0, mw, mh, 0, 0, embedding.inputSize, embedding.inputSize);
  const dst = new OffscreenCanvas(embedding.originalWidth, embedding.originalHeight);
  const dctx = dst.getContext('2d', { willReadFrequently: true })!;
  dctx.imageSmoothingEnabled = true;
  dctx.drawImage(crop, 0, 0, resizedW, resizedH, 0, 0, embedding.originalWidth, embedding.originalHeight);
  const rgba = dctx.getImageData(0, 0, embedding.originalWidth, embedding.originalHeight).data;
  const result = new Uint8Array(embedding.originalWidth * embedding.originalHeight);
  for (let i = 0; i < result.length; i++) result[i] = rgba[i * 4];
  return result;
}

export class MobileSamOnnxModel implements InteractiveSegmentationModel {
  private encoder?: ort.InferenceSession;
  private decoder?: ort.InferenceSession;

  async load() {
    [this.encoder, this.decoder] = await Promise.all([createSession(ENCODER_URL), createSession(DECODER_URL)]);
  }

  async encodeImage(image: ImageBitmap): Promise<ImageEmbedding> {
    if (!this.encoder) throw new Error('AI model is not loaded.');
    const { data, scale } = resizeImage(image, INPUT_SIZE);
    const input = new ort.Tensor('float32', toNchw(data, INPUT_SIZE), [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const feeds: Record<string, ort.Tensor> = { [this.encoder.inputNames[0]]: input };
    const output = await this.encoder.run(feeds);
    const tensor = output[this.encoder.outputNames[0]];
    return { tensor, originalWidth: image.width, originalHeight: image.height, inputSize: INPUT_SIZE, scale };
  }

  async predictMask({ embedding, points = [], box }: { embedding: ImageEmbedding; points?: PromptPoint[]; box?: BoundingBox }): Promise<SegmentationResult> {
    if (!this.decoder) throw new Error('AI model is not loaded.');
    const coords: number[] = [], labels: number[] = [];
    for (const p of points) { coords.push(p.x * embedding.scale, p.y * embedding.scale); labels.push(p.label); }
    if (box) {
      coords.push(box.x1 * embedding.scale, box.y1 * embedding.scale, box.x2 * embedding.scale, box.y2 * embedding.scale);
      labels.push(2, 3);
    }
    if (!labels.length) throw new Error('Add a point or bounding box prompt first.');
    const n = labels.length;
    const feeds: Record<string, ort.Tensor> = {};
    const find = (candidates: string[]) => candidates.find(n => this.decoder!.inputNames.includes(n));
    const embName = find(['image_embeddings', 'image_embedding']) || this.decoder.inputNames[0];
    const coordsName = find(['point_coords', 'point_coordinates']);
    const labelsName = find(['point_labels', 'point_label']);
    if (!coordsName || !labelsName) throw new Error(`Unsupported decoder inputs: ${this.decoder.inputNames.join(', ')}`);
    feeds[embName] = embedding.tensor;
    feeds[coordsName] = new ort.Tensor('float32', Float32Array.from(coords), [1, n, 2]);
    feeds[labelsName] = new ort.Tensor('float32', Float32Array.from(labels), [1, n]);
    if (this.decoder.inputNames.includes('mask_input')) feeds.mask_input = new ort.Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]);
    if (this.decoder.inputNames.includes('has_mask_input')) feeds.has_mask_input = new ort.Tensor('float32', Float32Array.of(0), [1]);
    if (this.decoder.inputNames.includes('orig_im_size')) feeds.orig_im_size = new ort.Tensor('float32', Float32Array.of(embedding.inputSize, embedding.inputSize), [2]);
    const out = await this.decoder.run(feeds);
    const maskTensor = out.masks || out.low_res_masks || out[this.decoder.outputNames[0]];
    const scoreTensor = out.iou_predictions || out.scores;
    const dims = maskTensor.dims.map(Number);
    const mh = dims[dims.length - 2], mw = dims[dims.length - 1];
    const all = maskTensor.data as Float32Array;
    const masks = Math.max(1, all.length / (mw * mh));
    let best = 0, bestScore = -Infinity;
    const scores = scoreTensor?.data as Float32Array | undefined;
    for (let i = 0; i < masks; i++) { const s = scores?.[i] ?? 0; if (s > bestScore) { bestScore = s; best = i; } }
    const slice = all.slice(best * mw * mh, (best + 1) * mw * mh);
    return { mask: maskToOriginal(slice, mw, mh, embedding), width: embedding.originalWidth, height: embedding.originalHeight, score: Number.isFinite(bestScore) ? bestScore : 0 };
  }
}
