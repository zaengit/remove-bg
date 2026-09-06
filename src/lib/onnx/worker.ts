import { MobileSamOnnxModel } from './model';
import type { BoundingBox, ImageEmbedding, PromptPoint } from '../../types/editor';

type Request =
  | { id: number; type: 'load' }
  | { id: number; type: 'encode'; image: ImageBitmap }
  | { id: number; type: 'predict'; workerKey: string; points?: PromptPoint[]; box?: BoundingBox };

type Success = { id: number; ok: true; result?: unknown };
type Failure = { id: number; ok: false; error: string };

const model = new MobileSamOnnxModel();
const embeddings = new Map<string, ImageEmbedding>();
let embeddingCounter = 0;

function fail(id: number, error: unknown): Failure {
  return {
    id,
    ok: false,
    error: error instanceof Error ? error.message : String(error),
  };
}

self.onmessage = async (event: MessageEvent<Request>) => {
  const message = event.data;
  try {
    if (message.type === 'load') {
      await model.load();
      self.postMessage({ id: message.id, ok: true } satisfies Success);
      return;
    }

    if (message.type === 'encode') {
      const embedding = await model.encodeImage(message.image);
      const workerKey = `embedding-${++embeddingCounter}`;
      embeddings.clear();
      embeddings.set(workerKey, embedding);
      message.image.close();
      self.postMessage({
        id: message.id,
        ok: true,
        result: {
          workerKey,
          originalWidth: embedding.originalWidth,
          originalHeight: embedding.originalHeight,
          inputSize: embedding.inputSize,
          scale: embedding.scale,
        },
      } satisfies Success);
      return;
    }

    const embedding = embeddings.get(message.workerKey);
    if (!embedding) throw new Error('Image embedding is no longer available. Re-upload the image.');
    const result = await model.predictMask({
      embedding,
      points: message.points,
      box: message.box,
    });
    self.postMessage(
      {
        id: message.id,
        ok: true,
        result: {
          ...result,
          mask: result.mask.buffer,
        },
      } satisfies Success,
      [result.mask.buffer],
    );
  } catch (error) {
    self.postMessage(fail(message.id, error));
  }
};
