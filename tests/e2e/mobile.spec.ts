import { expect, test } from '@playwright/test';

const png16x16 = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAHUlEQVR4nGP8z8Dwn4ECwESJ5lEDRg0YNWAwGQAAWG0CHvXMz6IAAAAASUVORK5CYII=',
  'base64',
);

test('mobile exposes remove and download actions', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');

  await page.locator('input[type="file"]').setInputFiles({
    name: 'sample.png',
    mimeType: 'image/png',
    buffer: png16x16,
  });

  await expect(page.getByRole('button', { name: 'Remove Selected' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Restore Selected' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Clear Selection' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Download PNG' })).toBeVisible();
});
