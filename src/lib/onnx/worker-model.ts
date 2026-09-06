import type {
  BoundingBox,
  ImageEmbedding,
  InteractiveSegmentationModel,
  PromptPoint,
  SegmentationResult,
} from '../../types/editor';

type Pending = {
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
};

type Response =
  | { id: number; ok: true; result?: unknown }
  | { id: number; ok: false; error: string };

export class WorkerSamOnnxModel implements InteractiveSegmentationModel {
  private worker: Worker;
  private requestId = 0;
  private pending = new Map<number, Pending>();

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<Response>) => {
      const message = event.data;
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.ok) pending.resolve(message.result);
      else pending.reject(new Error(message.error));
    };
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'AI worker crashed.');
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
    };
  }

  private request<T>(message: Record<string, unknown>, transfer: Transferable[] = []): Promise<T> {
    const id = ++this.requestId;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
      this.worker.postMessage({ id, ...message }, transfer);
    });
  }

  load(): Promise<void> {
    return this.request<void>({ type: 'load' });
  }

  encodeImage(image: ImageBitmap): Promise<ImageEmbedding> {
    return this.request<ImageEmbedding>({ type: 'encode', image }, [image]);
  }

  async predictMask({
    embedding,
    points,
    box,
  }: {
    embedding: ImageEmbedding;
    points?: PromptPoint[];
    box?: BoundingBox;
  }): Promise<SegmentationResult> {
    if (!embedding.workerKey) throw new Error('Worker embedding key is missing.');
    const result = await this.request<{
      mask: ArrayBuffer;
      width: number;
      height: number;
      score: number;
    }>({
      type: 'predict',
      workerKey: embedding.workerKey,
      points,
      box,
    });

    return {
      mask: new Uint8Array(result.mask),
      width: result.width,
      height: result.height,
      score: result.score,
    };
  }

  dispose() {
    this.worker.terminate();
    for (const pending of this.pending.values()) pending.reject(new Error('AI worker disposed.'));
    this.pending.clear();
  }
}
