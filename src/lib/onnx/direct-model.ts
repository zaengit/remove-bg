import * as ort from 'onnxruntime-web';
import type {
  BoundingBox,
  ImageEmbedding,
  InteractiveSegmentationModel,
  PromptPoint,
  SegmentationResult,
} from '../../types/editor';
import { samOnnxConfig } from './config';

async function createSession(url: string): Promise<ort.InferenceSession> {
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
  ort.env.wasm.simd = true;

  try {
    if ('gpu' in navigator) {
      return await ort.InferenceSession.create(url, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });
    }
  } catch (error) {
    console.warn('WebGPU ONNX failed; falling back to WASM.', error);
  }

  return ort.InferenceSession.create(url, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

function resolveName(
  available: readonly string[],
  explicit: string | undefined,
  candidates: string[],
  label: string,
  required = true,
): string | undefined {
  if (explicit) {
    if (!available.includes(explicit)) {
      throw new Error(`${label} '${explicit}' was configured but not found. Available: ${available.join(', ')}`);
    }
    return explicit;
  }
  const matched = candidates.find((candidate) => available.includes(candidate));
  if (matched) return matched;
  if (!required) return undefined;
  throw new Error(`Could not identify ${label}. Available: ${available.join(', ')}`);
}

function resizeImage(image: ImageBitmap, size: number) {
  const scale = size / Math.max(image.width, image.height);
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('OffscreenCanvas 2D context is unavailable.');
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(image, 0, 0, width, height);
  return { data: ctx.getImageData(0, 0, size, size).data, scale };
}

function toNchw(data: Uint8ClampedArray, size: number) {
  const out = new Float32Array(3 * size * size);
  const mean = [123.675, 116.28, 103.53];
  const std = [58.395, 57.12, 57.375];
  const plane = size * size;
  for (let i = 0; i < plane; i++) {
    out[i] = (data[i * 4] - mean[0]) / std[0];
    out[plane + i] = (data[i * 4 + 1] - mean[1]) / std[1];
    out[plane * 2 + i] = (data[i * 4 + 2] - mean[2]) / std[2];
  }
  return out;
}

function logitsToOriginal(values: Float32Array, maskWidth: number, maskHeight: number, embedding: ImageEmbedding): Uint8Array {
  const src = new OffscreenCanvas(maskWidth, maskHeight);
  const srcCtx = src.getContext('2d');
  if (!srcCtx) throw new Error('OffscreenCanvas 2D context is unavailable.');
  const image = srcCtx.createImageData(maskWidth, maskHeight);
  const pixelCount = maskWidth * maskHeight;
  for (let i = 0; i < pixelCount; i++) {
    const value = values[i] > 0 ? 255 : 0;
    image.data[i * 4] = value;
    image.data[i * 4 + 1] = value;
    image.data[i * 4 + 2] = value;
    image.data[i * 4 + 3] = 255;
  }
  srcCtx.putImageData(image, 0, 0);

  const resizedWidth = Math.max(1, Math.round(embedding.originalWidth * embedding.scale));
  const resizedHeight = Math.max(1, Math.round(embedding.originalHeight * embedding.scale));
  const cropWidth = Math.max(1, Math.min(maskWidth, Math.round(maskWidth * resizedWidth / embedding.inputSize)));
  const cropHeight = Math.max(1, Math.min(maskHeight, Math.round(maskHeight * resizedHeight / embedding.inputSize)));
  const dst = new OffscreenCanvas(embedding.originalWidth, embedding.originalHeight);
  const dstCtx = dst.getContext('2d', { willReadFrequently: true });
  if (!dstCtx) throw new Error('OffscreenCanvas 2D context is unavailable.');
  dstCtx.imageSmoothingEnabled = true;
  dstCtx.drawImage(src, 0, 0, cropWidth, cropHeight, 0, 0, embedding.originalWidth, embedding.originalHeight);
  const rgba = dstCtx.getImageData(0, 0, embedding.originalWidth, embedding.originalHeight).data;
  const result = new Uint8Array(embedding.originalWidth * embedding.originalHeight);
  for (let i = 0; i < result.length; i++) result[i] = rgba[i * 4];
  return result;
}

function tensorToFloat32(tensor: ort.Tensor): Float32Array {
  if (tensor.data instanceof Float32Array) return tensor.data;
  return Float32Array.from(tensor.data as ArrayLike<number>);
}

export class DirectMobileSamOnnxModel implements InteractiveSegmentationModel {
  private encoder?: ort.InferenceSession;
  private decoder?: ort.InferenceSession;

  async load() {
    [this.encoder, this.decoder] = await Promise.all([
      createSession(samOnnxConfig.encoderUrl),
      createSession(samOnnxConfig.decoderUrl),
    ]);
    console.info('SAM encoder contract', { inputs: this.encoder.inputNames, outputs: this.encoder.outputNames });
    console.info('SAM decoder contract', { inputs: this.decoder.inputNames, outputs: this.decoder.outputNames });
  }

  async encodeImage(image: ImageBitmap): Promise<ImageEmbedding> {
    if (!this.encoder) throw new Error('AI model is not loaded.');
    const inputName = resolveName(this.encoder.inputNames, samOnnxConfig.encoderInput, ['input_image', 'images', 'image', 'pixel_values'], 'encoder image input') ?? this.encoder.inputNames[0];
    const outputName = resolveName(this.encoder.outputNames, samOnnxConfig.encoderOutput, ['image_embeddings', 'image_embedding', 'embeddings'], 'encoder embedding output', false) ?? this.encoder.outputNames[0];
    const { data, scale } = resizeImage(image, samOnnxConfig.inputSize);
    const input = new ort.Tensor('float32', toNchw(data, samOnnxConfig.inputSize), [1, 3, samOnnxConfig.inputSize, samOnnxConfig.inputSize]);
    const output = await this.encoder.run({ [inputName]: input });
    const tensor = output[outputName];
    if (!tensor) throw new Error(`Encoder output '${outputName}' was not returned.`);
    return { tensor, originalWidth: image.width, originalHeight: image.height, inputSize: samOnnxConfig.inputSize, scale };
  }

  async predictMask({ embedding, points = [], box }: { embedding: ImageEmbedding; points?: PromptPoint[]; box?: BoundingBox }): Promise<SegmentationResult> {
    if (!this.decoder) throw new Error('AI model is not loaded.');
    if (!embedding.tensor) throw new Error('Direct ONNX embedding tensor is missing.');
    const coords: number[] = [];
    const labels: number[] = [];
    for (const point of points) { coords.push(point.x * embedding.scale, point.y * embedding.scale); labels.push(point.label); }
    if (box) { coords.push(box.x1 * embedding.scale, box.y1 * embedding.scale, box.x2 * embedding.scale, box.y2 * embedding.scale); labels.push(2, 3); }
    if (!labels.length) throw new Error('Add a point or bounding box prompt first.');

    const embeddingName = resolveName(this.decoder.inputNames, samOnnxConfig.decoderEmbeddingInput, ['image_embeddings', 'image_embedding', 'embeddings'], 'decoder embedding input')!;
    const coordsName = resolveName(this.decoder.inputNames, samOnnxConfig.decoderPointCoordsInput, ['point_coords', 'point_coordinates'], 'decoder point coordinates input')!;
    const labelsName = resolveName(this.decoder.inputNames, samOnnxConfig.decoderPointLabelsInput, ['point_labels', 'point_label'], 'decoder point labels input')!;
    const feeds: Record<string, ort.Tensor> = {
      [embeddingName]: embedding.tensor,
      [coordsName]: new ort.Tensor('float32', Float32Array.from(coords), [1, labels.length, 2]),
      [labelsName]: new ort.Tensor('float32', Float32Array.from(labels), [1, labels.length]),
    };

    const maskInputName = resolveName(this.decoder.inputNames, samOnnxConfig.decoderMaskInput, ['mask_input', 'mask_inputs'], 'decoder mask input', false);
    if (maskInputName) feeds[maskInputName] = new ort.Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]);
    const hasMaskName = resolveName(this.decoder.inputNames, samOnnxConfig.decoderHasMaskInput, ['has_mask_input', 'has_mask_inputs'], 'decoder has-mask input', false);
    if (hasMaskName) feeds[hasMaskName] = new ort.Tensor('float32', Float32Array.of(0), [1]);
    const originalSizeName = resolveName(this.decoder.inputNames, samOnnxConfig.decoderOriginalSizeInput, ['orig_im_size', 'original_size', 'orig_image_size'], 'decoder original-size input', false);
    if (originalSizeName) feeds[originalSizeName] = new ort.Tensor('float32', Float32Array.of(embedding.originalHeight, embedding.originalWidth), [2]);

    const outputs = await this.decoder.run(feeds);
    const maskOutputName = resolveName(this.decoder.outputNames, samOnnxConfig.decoderMaskOutput, ['low_res_masks', 'masks', 'mask'], 'decoder mask output', false) ?? this.decoder.outputNames[0];
    const scoreOutputName = resolveName(this.decoder.outputNames, samOnnxConfig.decoderScoreOutput, ['iou_predictions', 'scores', 'iou_scores'], 'decoder score output', false);
    const maskTensor = outputs[maskOutputName];
    if (!maskTensor) throw new Error(`Decoder mask output '${maskOutputName}' was not returned.`);
    const dims = maskTensor.dims.map(Number);
    if (dims.length < 2) throw new Error(`Unexpected mask tensor dimensions: [${dims.join(', ')}]`);
    const maskHeight = dims[dims.length - 2];
    const maskWidth = dims[dims.length - 1];
    const values = tensorToFloat32(maskTensor);
    const pixelsPerMask = maskWidth * maskHeight;
    const maskCount = Math.max(1, Math.floor(values.length / pixelsPerMask));
    const scores = scoreOutputName && outputs[scoreOutputName] ? tensorToFloat32(outputs[scoreOutputName]) : undefined;
    let bestIndex = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < maskCount; i++) {
      const score = scores?.[i] ?? 0;
      if (score > bestScore) { bestScore = score; bestIndex = i; }
    }
    const start = bestIndex * pixelsPerMask;
    const selected = values.slice(start, start + pixelsPerMask);
    return {
      mask: logitsToOriginal(selected, maskWidth, maskHeight, embedding),
      width: embedding.originalWidth,
      height: embedding.originalHeight,
      score: Number.isFinite(bestScore) ? bestScore : 0,
    };
  }
}
