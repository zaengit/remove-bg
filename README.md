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

Place a browser-compatible split SAM-style ONNX model at:

```text
public/models/mobile_sam_encoder.onnx
public/models/mobile_sam_decoder.onnx
```

The adapter supports the common MobileSAM/SAM contract:

- encoder image input such as `input_image`, `images`, `image`, or `pixel_values`
- encoder embedding output such as `image_embeddings`, `image_embedding`, or `embeddings`
- decoder embedding input such as `image_embeddings`
- decoder prompt inputs such as `point_coords` and `point_labels`
- optional `mask_input`, `has_mask_input`, and `orig_im_size`
- mask outputs such as `low_res_masks`, `masks`, or `mask`
- score outputs such as `iou_predictions`, `scores`, or `iou_scores`

The image is resized with aspect ratio preserved and padded on the bottom/right to the configured square input size. Low-resolution mask output is cropped back to the unpadded image region before it is resized to original resolution.

### Model configuration

Basic configuration:

```bash
VITE_ENCODER_MODEL_URL=/models/mobile_sam_encoder.onnx
VITE_DECODER_MODEL_URL=/models/mobile_sam_decoder.onnx
VITE_MODEL_INPUT_SIZE=1024
```

If your ONNX export uses different tensor names, configure them without modifying application code:

```bash
VITE_ENCODER_INPUT_NAME=input_image
VITE_ENCODER_OUTPUT_NAME=image_embeddings

VITE_DECODER_EMBEDDING_INPUT_NAME=image_embeddings
VITE_DECODER_POINT_COORDS_INPUT_NAME=point_coords
VITE_DECODER_POINT_LABELS_INPUT_NAME=point_labels
VITE_DECODER_MASK_INPUT_NAME=mask_input
VITE_DECODER_HAS_MASK_INPUT_NAME=has_mask_input
VITE_DECODER_ORIGINAL_SIZE_INPUT_NAME=orig_im_size

VITE_DECODER_MASK_OUTPUT_NAME=low_res_masks
VITE_DECODER_SCORE_OUTPUT_NAME=iou_predictions
```

All overrides are optional. The runtime attempts to identify common tensor names automatically and logs the detected encoder/decoder input and output contracts to the browser console after model loading. If a configured name does not exist, the editor returns a descriptive error listing available tensor names.

Model-specific preprocessing is isolated in `src/lib/onnx/model.ts`, while paths and tensor-name configuration live in `src/lib/onnx/config.ts`. A different segmentation architecture can also implement the `InteractiveSegmentationModel` interface without changing the editor.

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
