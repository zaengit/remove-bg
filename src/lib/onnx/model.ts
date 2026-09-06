import type {
  BoundingBox,
  ImageEmbedding,
  InteractiveSegmentationModel,
  PromptPoint,
  SegmentationResult,
} from '../../types/editor';
import { DirectMobileSamOnnxModel } from './direct-model';
import { WorkerSamOnnxModel } from './worker-model';

/**
 * Public SAM model facade.
 *
 * Browser main-thread usage prefers a dedicated Web Worker so preprocessing,
 * encoder inference, and decoder inference do not block editor rendering.
 * Environments without Worker support fall back to direct ONNX execution.
 */
export class MobileSamOnnxModel implements InteractiveSegmentationModel {
  private implementation: InteractiveSegmentationModel;

  constructor() {
    this.implementation = typeof Worker !== 'undefined'
      ? new WorkerSamOnnxModel()
      : new DirectMobileSamOnnxModel();
  }

  load(): Promise<void> {
    return this.implementation.load();
  }

  encodeImage(image: ImageBitmap): Promise<ImageEmbedding> {
    return this.implementation.encodeImage(image);
  }

  predictMask(input: {
    embedding: ImageEmbedding;
    points?: PromptPoint[];
    box?: BoundingBox;
  }): Promise<SegmentationResult> {
    return this.implementation.predictMask(input);
  }
}
