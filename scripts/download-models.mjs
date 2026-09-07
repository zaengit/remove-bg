import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

const MODEL_SHA256 = '7bad6522b3cde60246e69e234b7786337ef9c88abc790ee5c1aaa6e535b0c61d';
const MODEL_VERSION = MODEL_SHA256.slice(0, 8);

const model = {
  url: 'https://huggingface.co/Xenova/modnet/resolve/main/onnx/model_uint8.onnx?download=true',
  path: `public/models/modnet-uint8-${MODEL_VERSION}.onnx`,
  sha256: MODEL_SHA256,
};

console.log(`Downloading ${model.path}...`);
const response = await fetch(model.url, { redirect: 'follow' });
if (!response.ok) throw new Error(`Failed to download ${model.url}: ${response.status} ${response.statusText}`);

const bytes = Buffer.from(await response.arrayBuffer());
const digest = createHash('sha256').update(bytes).digest('hex');
if (digest !== model.sha256) {
  throw new Error(`Checksum mismatch for ${model.path}: expected ${model.sha256}, got ${digest}`);
}

await mkdir(dirname(model.path), { recursive: true });
await writeFile(model.path, bytes);
console.log(`Verified ${model.path} (${(bytes.length / 1024 / 1024).toFixed(2)} MiB)`);
console.log('MODNet uint8 model is ready with a content-versioned filename.');
