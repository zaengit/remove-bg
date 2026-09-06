import { copyFile, mkdir, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';

const sourceDir = resolve('node_modules/onnxruntime-web/dist');
const targetDir = resolve('public/ort');

await mkdir(targetDir, { recursive: true });

const files = (await readdir(sourceDir)).filter(
  (name) => name.startsWith('ort-wasm') && (name.endsWith('.wasm') || name.endsWith('.mjs')),
);

if (!files.some((name) => name.endsWith('.wasm'))) {
  throw new Error(`No ONNX Runtime WASM binaries found in ${sourceDir}`);
}

for (const name of files) {
  await copyFile(resolve(sourceDir, name), resolve(targetDir, name));
}

console.log(`Copied ${files.length} ONNX Runtime web runtime files to public/ort/.`);
