import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import { ImageEditor } from './components/editor/ImageEditor';

function MobileRemoveAction() {
  const removeSelected = () => {
    const button = Array.from(document.querySelectorAll<HTMLButtonElement>('button'))
      .find((candidate) => candidate.textContent?.trim() === 'Remove Selected');
    button?.click();
  };

  return (
    <button
      type="button"
      onClick={removeSelected}
      className="fixed bottom-24 left-1/2 z-50 -translate-x-1/2 rounded-2xl bg-blue-600 px-6 py-3 text-sm font-semibold text-white shadow-2xl shadow-black/40 active:scale-95 lg:hidden"
      aria-label="Remove Selected"
    >
      Remove Selected
    </button>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ImageEditor />
    <MobileRemoveAction />
  </React.StrictMode>
);
