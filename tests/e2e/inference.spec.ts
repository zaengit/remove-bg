import { expect, test } from '@playwright/test';

async function uploadSyntheticSquare(page: import('@playwright/test').Page) {
  await page.locator('input[type="file"]').evaluate(async (input) => {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context unavailable in test.');

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, 256, 256);
    ctx.fillStyle = '#000000';
    ctx.fillRect(64, 64, 128, 128);

    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((value) => value ? resolve(value) : reject(new Error('PNG encoding failed.')), 'image/png');
    });

    const file = new File([blob], 'synthetic-square.png', { type: 'image/png' });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    const fileInput = input as HTMLInputElement;
    fileInput.files = transfer.files;
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

test.describe('MobileSAM real inference', () => {
  test.setTimeout(180_000);

  test('click prompt selects and removes a simple object', async ({ page }) => {
    const pageErrors: Error[] = [];
    page.on('pageerror', (error) => pageErrors.push(error));

    await page.goto('/');
    await expect(page.getByText('AI ready')).toBeVisible({ timeout: 120_000 });

    await uploadSyntheticSquare(page);
    await expect(page.getByText('Ready — Smart Select an object')).toBeVisible({ timeout: 120_000 });

    const canvas = page.locator('canvas');
    const bounds = await canvas.boundingBox();
    expect(bounds).not.toBeNull();
    if (!bounds) return;

    await page.mouse.click(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    await expect(page.getByText(/Selected · confidence/)).toBeVisible({ timeout: 60_000 });

    await page.getByRole('button', { name: 'Remove Selected' }).click();
    await page.getByRole('button', { name: 'Clear Selection' }).click();

    const centerAlpha = await canvas.evaluate((node) => {
      const element = node as HTMLCanvasElement;
      const ctx = element.getContext('2d');
      if (!ctx) return -1;
      return ctx.getImageData(Math.floor(element.width / 2), Math.floor(element.height / 2), 1, 1).data[3];
    });

    expect(centerAlpha).toBe(0);
    expect(pageErrors).toEqual([]);
  });
});
