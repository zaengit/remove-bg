import { useEffect, useRef, useState } from 'react';
import { Download, Sparkles, Upload } from 'lucide-react';
import { BackgroundRemovalOnnxModel } from '../../lib/onnx/background-model';

const model = new BackgroundRemovalOnnxModel();

type ResultStats = {
  transparentRatio: number;
  opaqueRatio: number;
  softRatio: number;
};

export function ImageEditor() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const bitmapRef = useRef<ImageBitmap | null>(null);
  const originalRef = useRef<ImageData | null>(null);
  const currentRef = useRef<ImageData | null>(null);

  const [status, setStatus] = useState('Loading AI model...');
  const [error, setError] = useState<string | null>(null);
  const [hasImage, setHasImage] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [hasResult, setHasResult] = useState(false);
  const [resultStats, setResultStats] = useState<ResultStats | null>(null);
  const [version, setVersion] = useState(0);

  useEffect(() => {
    model
      .load()
      .then(() => {
        setIsModelReady(true);
        setStatus(`AI ready · ${model.activeBackend.toUpperCase()}`);
      })
      .catch((e: Error) => {
        setStatus('AI unavailable');
        setError(`Model loading failed: ${e.message}`);
      });
  }, []);

  const draw = () => {
    const canvas = canvasRef.current;
    const host = hostRef.current;
    const image = currentRef.current;
    if (!canvas || !host) return;

    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, host.clientWidth);
    const height = Math.max(1, host.clientHeight);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!image) return;

    const source = document.createElement('canvas');
    source.width = image.width;
    source.height = image.height;
    source.getContext('2d')?.putImageData(image, 0, 0);

    const scale = Math.min((width - 48) / image.width, (height - 48) / image.height, 1);
    const renderWidth = image.width * scale;
    const renderHeight = image.height * scale;
    ctx.drawImage(source, (width - renderWidth) / 2, (height - renderHeight) / 2, renderWidth, renderHeight);
  };

  useEffect(draw, [version]);
  useEffect(() => {
    const onResize = () => draw();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [version]);

  const loadImage = async (file: File) => {
    setError(null);
    setHasResult(false);
    setResultStats(null);

    if (file.size > 50 * 1024 * 1024) {
      setError('Image is too large (50 MB maximum).');
      return;
    }

    try {
      bitmapRef.current?.close();
      const bitmap = await createImageBitmap(file);
      bitmapRef.current = bitmap;

      const offscreen = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = offscreen.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw new Error('Canvas is not available.');
      ctx.drawImage(bitmap, 0, 0);
      const data = ctx.getImageData(0, 0, bitmap.width, bitmap.height);

      originalRef.current = new ImageData(new Uint8ClampedArray(data.data), data.width, data.height);
      currentRef.current = new ImageData(new Uint8ClampedArray(data.data), data.width, data.height);
      setHasImage(true);
      setVersion((v) => v + 1);
      setStatus(isModelReady ? 'Ready to remove background' : 'Waiting for AI model...');
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(`Image loading failed: ${message}`);
    }
  };

  const removeBackground = async () => {
    const bitmap = bitmapRef.current;
    const original = originalRef.current;
    if (!bitmap || !original || !isModelReady || isProcessing) return;

    setIsProcessing(true);
    setError(null);
    setStatus('Removing background...');

    try {
      const mask = await model.removeBackground(bitmap);
      const output = new ImageData(new Uint8ClampedArray(original.data), original.width, original.height);
      let transparent = 0;
      let opaque = 0;
      let soft = 0;

      for (let i = 0; i < mask.length; i++) {
        const originalAlpha = original.data[i * 4 + 3];
        const alpha = Math.round((originalAlpha * mask[i]) / 255);
        output.data[i * 4 + 3] = alpha;
        if (alpha <= 16) transparent++;
        else if (alpha >= 239) opaque++;
        else soft++;
      }

      const total = Math.max(1, mask.length);
      setResultStats({
        transparentRatio: transparent / total,
        opaqueRatio: opaque / total,
        softRatio: soft / total,
      });
      currentRef.current = output;
      setHasResult(true);
      setStatus(`Background removed · ${model.activeBackend.toUpperCase()}`);
      setVersion((v) => v + 1);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(`Background removal failed: ${message}`);
      setStatus('Ready to try again');
    } finally {
      setIsProcessing(false);
    }
  };

  const download = () => {
    const image = currentRef.current;
    if (!image || !hasResult) return;

    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    canvas.getContext('2d')?.putImageData(image, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'removed-background.png';
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, 'image/png');
  };

  const canRemove = hasImage && isModelReady && !isProcessing;

  return (
    <div className="flex h-full flex-col bg-neutral-950 text-neutral-100">
      <header className="flex min-h-14 shrink-0 flex-wrap items-center gap-2 border-b border-white/10 px-3 py-2">
        <div className="mr-2 flex items-center gap-2 font-semibold">
          <span className="grid size-8 place-items-center rounded-lg bg-blue-600"><Sparkles size={17} /></span>
          Remove BG
        </div>

        <label className="flex cursor-pointer items-center gap-2 rounded-lg bg-white px-3 py-2 text-sm font-medium text-neutral-900">
          <Upload size={16} /> Upload
          <input type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && loadImage(e.target.files[0])} />
        </label>

        <div className="ml-auto flex items-center gap-2 text-xs text-neutral-400">
          <span className="hidden sm:inline">Your images stay on your device.</span>
          <span className="rounded-full bg-white/5 px-2 py-1">{status}</span>
        </div>
      </header>

      <main className="flex min-h-0 flex-1 flex-col">
        <section ref={hostRef} className="checkerboard relative min-h-0 flex-1 overflow-hidden">
          <canvas
            ref={canvasRef}
            className="absolute inset-0"
            data-transparent-ratio={resultStats?.transparentRatio.toFixed(4) ?? ''}
            data-opaque-ratio={resultStats?.opaqueRatio.toFixed(4) ?? ''}
            data-soft-ratio={resultStats?.softRatio.toFixed(4) ?? ''}
          />

          {!hasImage && (
            <div className="pointer-events-none absolute inset-0 grid place-items-center p-6">
              <div className="rounded-2xl border border-white/10 bg-neutral-900/90 p-8 text-center shadow-2xl">
                <Upload className="mx-auto mb-3 text-blue-400" />
                <h2 className="text-lg font-semibold">Upload an image to start</h2>
                <p className="mt-1 text-sm text-neutral-400">AI automatically keeps the foreground and removes the background.</p>
              </div>
            </div>
          )}

          {error && (
            <div className="absolute left-1/2 top-4 max-w-[calc(100%-2rem)] -translate-x-1/2 rounded-lg border border-red-500/30 bg-red-950/90 px-4 py-2 text-sm text-red-200">
              {error}
            </div>
          )}
        </section>

        <div className="flex shrink-0 flex-wrap items-center justify-center gap-3 border-t border-white/10 bg-neutral-950 p-4">
          <button
            type="button"
            onClick={removeBackground}
            disabled={!canRemove}
            className="min-w-44 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isProcessing ? 'Removing...' : 'Remove Background'}
          </button>
          <button
            type="button"
            onClick={download}
            disabled={!hasResult}
            className="flex min-w-44 items-center justify-center gap-2 rounded-xl border border-white/10 px-5 py-3 text-sm font-medium hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Download size={16} /> Download PNG
          </button>
        </div>
      </main>
    </div>
  );
}
