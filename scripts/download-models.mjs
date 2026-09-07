import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

const BASE = 'https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx';

const models = [
  {
    url: `${BASE}/model_fp16.onnx?download=true`,
    path: 'public/models/rmbg-1.4-fp16.onnx',
    sha256: '9fdfdb41866d872e0acf4a010c35c1a8547bf0eebe0d1544406bbf1c824cb59d',
  },
  {
    url: `${BASE}/model_quantized.onnx?download=true`,
    path: 'public/models/rmbg-1.4-quantized.onnx',
    sha256: 'a6648479275dfd0ede0f3a8abc20aa5c437b394681b05e5af6d268250aaf40f3',
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

console.log('RMBG 1.4 models are ready.');
