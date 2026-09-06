# Remove BG

Private, fully client-side interactive background removal using React, TypeScript, ONNX Runtime Web, WebGPU/WASM execution, Canvas, and Rust WebAssembly.

## What works

- Image upload via `ImageBitmap`
- MobileSAM/SAM-style ONNX encoder + decoder abstraction
- Encoder runs once per uploaded image and caches its embedding
- WebGPU execution first, automatic ONNX Runtime WASM fallback
- Positive click, Alt/Option negative click, and drag bounding-box prompts
- Replace/add/subtract AI selections
- Brush and eraser mask refinement
- Remove selected pixels by setting alpha to zero
- Restore selected pixels from the immutable original image
- Undo/redo for mask and alpha state
- Canvas pan, zoom, fit-to-screen, DPR-aware rendering, checkerboard transparency
- Transparent PNG export
- Rust/WASM mask union/subtract, threshold, resize, morphology, feather, remove/restore helpers
- No backend and no image upload

## Prerequisites

- Node.js 20+
- Rust stable
- `wasm-pack` (`cargo install wasm-pack`)
- A browser with WebAssembly; WebGPU is recommended

## Model files

Place a browser-compatible SAM-style split ONNX model at:

```text
public/models/mobile_sam_encoder.onnx
public/models/mobile_sam_decoder.onnx
```

The default adapter expects a MobileSAM/SAM decoder using common inputs such as `image_embeddings`, `point_coords`, `point_labels`, optional `mask_input`, `has_mask_input`, and `orig_im_size`, and outputs `masks`/`low_res_masks` plus `iou_predictions`/`scores`.

Model exports differ. If yours uses different tensor names or preprocessing, update only `src/lib/onnx/model.ts` or provide a second `InteractiveSegmentationModel` implementation.

Optional environment configuration:

```bash
VITE_ENCODER_MODEL_URL=/models/mobile_sam_encoder.onnx
VITE_DECODER_MODEL_URL=/models/mobile_sam_decoder.onnx
VITE_MODEL_INPUT_SIZE=1024
```

## Development

```bash
npm install
npm run wasm:build:dev
npm run dev
```

Open the Vite URL, upload an image, then use Smart Select:

- Click: positive point
- Alt/Option + click: negative point
- Drag: bounding box
- Shift + click: add AI result to current selection
- Add Selection tool: union the next AI result
- Subtract tool: subtract the next AI result

## Production build

```bash
npm install
npm run build
npm run preview
```

`npm run build` compiles Rust using `wasm-pack`, type-checks TypeScript, and builds the Vite bundle.

## Privacy

**Your images stay on your device.** Image pixels and ONNX inference remain inside the browser. The application has no image-upload backend.

## Notes for production deployment

Serve the site with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` if you want threaded ONNX Runtime WASM. The Vite development server already sets these headers. Configure equivalent headers on your CDN/host.

For very large images, add a dedicated worker pipeline for model preprocessing and expensive mask post-processing; the current version already avoids storing the full-resolution image in React state and keeps image buffers in refs.
