import { expect, test } from '@playwright/test';

const syntheticSquare = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAAC0UlEQVR4nO3VsQ0DAQzDwDjI/iv7d/gULng3gRpCs7sfqPpeD4BLAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgRA2u96wL9m5npC3e5eT3jPA5AmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYC02d3rDXDGA5AmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIg7QHrVgr7LPh0aAAAAABJRU5ErkJggg==',
  'base64',
);

test.describe('MobileSAM real inference', () => {
  test.setTimeout(180_000);

  test('click prompt selects and removes a simple object', async ({ page }) => {
    const pageErrors: Error[] = [];
    page.on('pageerror', (error) => pageErrors.push(error));

    await page.goto('/');
    await expect(page.getByText('AI ready')).toBeVisible({ timeout: 120_000 });

    await page.locator('input[type="file"]').setInputFiles({
      name: 'synthetic-square.png',
      mimeType: 'image/png',
      buffer: syntheticSquare,
    });

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
