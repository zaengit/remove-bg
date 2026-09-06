import { expect, test } from '@playwright/test';

const png16x16 = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAHUlEQVR4nGP8z8Dwn4ECwESJ5lEDRg0YNWAwGQAAWG0CHvXMz6IAAAAASUVORK5CYII=',
  'base64',
);

test.describe('Remove BG editor', () => {
  test('renders editor shell and privacy message', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByText('Remove BG', { exact: true })).toBeVisible();
    await expect(page.getByText('Your images stay on your device.')).toBeVisible();
    await expect(page.getByText('Upload an image to start')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Remove Selected' })).toBeVisible();
  });

  test('fails gracefully when ONNX model files are absent', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByText('AI unavailable')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/Model loading failed:/)).toBeVisible();
  });

  test('uploads an image and keeps manual editing controls usable without AI', async ({ page }) => {
    const pageErrors: Error[] = [];
    page.on('pageerror', (error) => pageErrors.push(error));

    await page.goto('/');
    await page.locator('input[type="file"]').setInputFiles({
      name: 'sample.png',
      mimeType: 'image/png',
      buffer: png16x16,
    });

    await expect(page.getByText('Upload an image to start')).toBeHidden();
    await expect(page.getByRole('button', { name: 'Fit' })).toBeVisible();

    await page.getByTitle('Brush').click();
    const canvas = page.locator('canvas');
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();

    if (box) {
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width / 2 + 2, box.y + box.height / 2 + 2);
      await page.mouse.up();
    }

    await page.getByRole('button', { name: 'Remove Selected' }).click();
    await page.getByTitle('Undo').click();
    await page.getByTitle('Redo').click();

    expect(pageErrors).toEqual([]);
  });

  test('zooms and exports a PNG', async ({ page }) => {
    await page.goto('/');
    await page.locator('input[type="file"]').setInputFiles({
      name: 'sample.png',
      mimeType: 'image/png',
      buffer: png16x16,
    });

    await expect(page.getByRole('button', { name: 'Fit' })).toBeVisible();
    await expect(page.getByText('100%')).toBeVisible();

    const zoomIn = page.locator('section button').filter({ has: page.locator('svg') }).nth(1);
    await zoomIn.click();
    await expect(page.getByText('120%')).toBeVisible();

    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download PNG' }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe('removed-background.png');
    const stream = await download.createReadStream();
    const chunks: Buffer[] = [];
    for await (const chunk of stream) chunks.push(Buffer.from(chunk));
    expect(Buffer.concat(chunks).length).toBeGreaterThan(20);
  });
});
