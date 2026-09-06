export type Tool = 'move' | 'smart' | 'add' | 'subtract' | 'brush' | 'eraser';
export interface PromptPoint { x: number; y: number; label: 0 | 1; }
export interface BoundingBox { x1: number; y1: number; x2: number; y2: number; }
export interface ImageEmbedding {
  tensor: import('onnxruntime-web').Tensor;
  originalWidth: number;
  originalHeight: number;
  inputSize: number;
  scale: number;
}
export interface SegmentationResult { mask: Uint8Array; width: number; height: number; score: number; }
export interface InteractiveSegmentationModel {
  load(): Promise<void>;
  encodeImage(image: ImageBitmap): Promise<ImageEmbedding>;
  predictMask(input: { embedding: ImageEmbedding; points?: PromptPoint[]; box?: BoundingBox }): Promise<SegmentationResult>;
}
