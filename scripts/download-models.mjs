import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

const REVISION = '0d3b403339b4674a82493d5e97964dd78089ddc8';
const BASE = `https://huggingface.co/Acly/MobileSAM/resolve/${REVISION}`;

const models = [
  {
    url: `${BASE}/mobile_sam_image_encoder.onnx?download=true`,
    path: 'public/models/mobile_sam_encoder.onnx',
    sha256: '580f5fb648ea1062c0aabc26217aed56921985f03f0cbbd852bba81d760cc749',
  },
  {
    url: `${BASE}/sam_mask_decoder_single.onnx?download=true`,
    path: 'public/models/mobile_sam_decoder.onnx',
    sha256: '93915fc7c993ab9d59ab8c9ccd3bce37f7509c81ab4150a74abd4d2abbd8570d',
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

console.log('MobileSAM models are ready.');
