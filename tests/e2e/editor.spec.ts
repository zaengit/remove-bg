import { expect, test } from '@playwright/test';

const png16x16 = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAHUlEQVR4nGP8z8Dwn4ECwESJ5lEDRg0YNWAwGQAAWG0CHvXMz6IAAAAASUVORK5CYII=',
  'base64',
);

test.describe('Remove BG editor', () => {
  test('renders automatic removal controls', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByText('Remove BG', { exact: true })).toBeVisible();
    await expect(page.getByText('Your images stay on your device.')).toBeVisible();
    await expect(page.getByText('Upload an image to start')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Remove Background' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Download PNG' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Remove Selected' })).toHaveCount(0);
    await expect(page.getByTitle('Brush')).toHaveCount(0);
  });

  test('shows unavailable state when local ONNX assets are absent', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByText('AI unavailable')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/Model loading failed:/)).toBeVisible();
  });

  test('uploads an image without selection controls', async ({ page }) => {
    const pageErrors: Error[] = [];
    page.on('pageerror', (error) => pageErrors.push(error));

    await page.goto('/');
    await page.locator('input[type="file"]').setInputFiles({
      name: 'sample.png',
      mimeType: 'image/png',
      buffer: png16x16,
    });

    await expect(page.getByText('Upload an image to start')).toBeHidden();
    await expect(page.getByRole('button', { name: 'Download PNG' })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Restore Selected' })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Clear Selection' })).toHaveCount(0);
    expect(pageErrors).toEqual([]);
  });
});
