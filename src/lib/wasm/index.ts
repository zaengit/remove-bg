let api: any = null;
export async function loadMaskWasm() {
  if (api) return api;
  try {
    const url = new URL('./pkg/remove_bg_wasm.js', import.meta.url).href;
    const mod = await import(/* @vite-ignore */ url);
    await mod.default(); api = mod; return api;
  } catch (e) { console.warn('Rust WASM unavailable; using JS mask fallback.', e); return null; }
}
export function unionMasks(a: Uint8Array, b: Uint8Array) { const o = new Uint8Array(a.length); for (let i=0;i<o.length;i++) o[i]=Math.max(a[i],b[i]); return o; }
export function subtractMasks(a: Uint8Array, b: Uint8Array) { const o = new Uint8Array(a.length); for (let i=0;i<o.length;i++) o[i]=b[i]>127?0:a[i]; return o; }
export async function unionMasksFast(a: Uint8Array,b: Uint8Array){ const w=await loadMaskWasm(); return w?new Uint8Array(w.union_masks(a,b)):unionMasks(a,b); }
export async function subtractMasksFast(a: Uint8Array,b: Uint8Array){ const w=await loadMaskWasm(); return w?new Uint8Array(w.subtract_masks(a,b)):subtractMasks(a,b); }
