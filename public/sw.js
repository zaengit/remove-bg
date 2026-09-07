const MODEL_VERSION = '309c8469';
const MODEL_CACHE = `remove-bg-models-${MODEL_VERSION}`;
const MODEL_FILE = `/models/u2netp-${MODEL_VERSION}.onnx`;

self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys
        .filter((key) => key.startsWith('remove-bg-models-') && key !== MODEL_CACHE)
        .map((key) => caches.delete(key)),
    );
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin || url.pathname !== MODEL_FILE) return;

  event.respondWith((async () => {
    const cache = await caches.open(MODEL_CACHE);
    const cached = await cache.match(request);
    if (cached) return cached;

    const response = await fetch(request);
    if (response.ok) await cache.put(request, response.clone());
    return response;
  })());
});
