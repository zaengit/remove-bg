const MODEL_VERSION = '309c8469';
const APP_VERSION = 'pwa-v1';
const MODEL_CACHE = `remove-bg-models-${MODEL_VERSION}`;
const APP_CACHE = `remove-bg-app-${APP_VERSION}`;
const MODEL_FILE = `/models/u2netp-${MODEL_VERSION}.onnx`;
const APP_SHELL = ['/', '/manifest.webmanifest', '/icon.svg'];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(APP_CACHE);
    await cache.addAll(APP_SHELL);
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys
        .filter((key) =>
          (key.startsWith('remove-bg-models-') && key !== MODEL_CACHE) ||
          (key.startsWith('remove-bg-app-') && key !== APP_CACHE),
        )
        .map((key) => caches.delete(key)),
    );
    await self.clients.claim();
  })());
});

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) await cache.put(request, response.clone());
  return response;
}

async function networkFirstNavigation(request) {
  const cache = await caches.open(APP_CACHE);
  try {
    const response = await fetch(request);
    if (response.ok) await cache.put('/', response.clone());
    return response;
  } catch {
    return (await cache.match('/')) || Response.error();
  }
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (url.pathname === MODEL_FILE) {
    event.respondWith(cacheFirst(request, MODEL_CACHE));
    return;
  }

  if (request.mode === 'navigate') {
    event.respondWith(networkFirstNavigation(request));
    return;
  }

  if (['script', 'style', 'image', 'font'].includes(request.destination) || url.pathname.endsWith('.webmanifest')) {
    event.respondWith(cacheFirst(request, APP_CACHE));
  }
});
