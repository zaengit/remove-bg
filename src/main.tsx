import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import { ImageEditor } from './components/editor/ImageEditor';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><ImageEditor /></React.StrictMode>
);
