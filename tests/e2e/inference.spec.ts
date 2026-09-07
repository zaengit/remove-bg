import { expect, test } from '@playwright/test';

async function uploadSyntheticSubject(page: import('@playwright/test').Page) {
  await page.locator('input[type="file"]').evaluate(async (input) => {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context unavailable in test.');

    const gradient = ctx.createLinearGradient(0, 0, 640, 480);
    gradient.addColorStop(0, '#6840b8');
    gradient.addColorStop(1, '#14224a');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 640, 480);

    // Stage-like background details that should not all become soft foreground.
    ctx.fillStyle = 'rgba(255,255,255,.65)';
    ctx.font = 'bold 34px sans-serif';
    ctx.fillText('KEYNOTE ADDRESS', 300, 75);
    for (let y = 330; y < 450; y += 45) {
      for (let x = 30; x < 620; x += 120) ctx.fillText('○', x, y);
    }

    // Simple central person/podium silhouette.
    ctx.fillStyle = '#d7b08c';
    ctx.beginPath();
    ctx.arc(245, 190, 38, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#15171d';
    ctx.beginPath();
    ctx.moveTo(195, 235);
    ctx.lineTo(290, 235);
    ctx.lineTo(315, 390);
    ctx.lineTo(170, 390);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = '#3a2a20';
    ctx.fillRect(130, 340, 245, 88);

    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((value) => value ? resolve(value) : reject(new Error('PNG encoding failed.')), 'image/png');
    });

    const file = new File([blob], 'synthetic-stage.png', { type: 'image/png' });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    const fileInput = input as HTMLInputElement;
    fileInput.files = transfer.files;
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

test.describe('U2NetP real inference', () => {
  test.setTimeout(180_000);

  test('removes background with cleaned alpha and exports PNG', async ({ page }) => {
    const pageErrors: Error[] = [];
    page.on('pageerror', (error) => pageErrors.push(error));

    await page.goto('/');
    await expect(page.getByText(/AI ready · WASM/)).toBeVisible({ timeout: 120_000 });

    await uploadSyntheticSubject(page);
    await expect(page.getByText('Ready to remove background')).toBeVisible();

    await page.getByRole('button', { name: 'Remove Background' }).click();
    await expect(page.getByText(/Background removed · WASM/)).toBeVisible({ timeout: 120_000 });
    await expect(page.getByRole('button', { name: 'Download PNG' })).toBeEnabled();

    const canvas = page.locator('canvas');
    const stats = await canvas.evaluate((node) => ({
      transparent: Number(node.dataset.transparentRatio),
      opaque: Number(node.dataset.opaqueRatio),
      soft: Number(node.dataset.softRatio),
    }));

    expect(stats.transparent).toBeGreaterThan(0.05);
    expect(stats.opaque).toBeGreaterThan(0.005);
    // Regression guard for the previous ghosted result where most pixels were
    // left semi-transparent instead of becoming clear foreground/background.
    expect(stats.soft).toBeLessThan(0.65);

    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download PNG' }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe('removed-background.png');
    expect(pageErrors).toEqual([]);
  });

  test('stores the versioned model in Cache Storage', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText(/AI ready · WASM/)).toBeVisible({ timeout: 120_000 });

    const cached = await page.evaluate(async () => {
      const keys = await caches.keys();
      const key = keys.find((value) => value === 'remove-bg-models-309c8469');
      if (!key) return false;
      const cache = await caches.open(key);
      const requests = await cache.keys();
      return requests.some((request) => request.url.includes('u2netp-309c8469.onnx'));
    });

    expect(cached).toBe(true);
  });
});
