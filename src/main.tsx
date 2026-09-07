import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import { ImageEditor } from './components/editor/ImageEditor';

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    const serviceWorkerUrl = new URL('sw.js', new URL(import.meta.env.BASE_URL, window.location.origin));
    navigator.serviceWorker.register(serviceWorkerUrl.href).catch((error) => {
      console.warn('Service worker registration failed.', error);
    });
  });
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ImageEditor />
  </React.StrictMode>
);
