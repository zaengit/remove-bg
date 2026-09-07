import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

const BASE = 'https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx';

const models = [
  {
    url: `${BASE}/model_fp16.onnx?download=true`,
    path: 'public/models/birefnet-lite-fp16.onnx',
    sha256: 'd39b897ceb16ae654c1731f3dba0cf9b368d9cae74b5a57459b455cc8bfec402',
  },
  {
    url: `${BASE}/model.onnx?download=true`,
    path: 'public/models/birefnet-lite-fp32.onnx',
    sha256: '5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333',
  },
];

for (const model of models) {
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
  console.log(`Verified ${model.path} (${(bytes.length / 1024 / 1024).toFixed(1)} MB)`);
}

console.log('BiRefNet Lite models are ready.');
