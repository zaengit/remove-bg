import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

const model = {
  url: 'https://huggingface.co/edgetools/u2netp/resolve/25dee37/u2netp.onnx?download=true',
  path: 'public/models/u2netp.onnx',
  sha256: '309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8',
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
console.log('U2NetP model is ready.');
