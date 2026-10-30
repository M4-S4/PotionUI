import { test, expect } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';

const JOURNEY = 'prompts-grid-phone';
const VIEWPORT_WIDTH = 390;
const GUTTER = 16;

test('prompt cards fit the phone viewport with the usage meta visible', async ({ page }) => {
	test.setTimeout(60000);
	await page.setViewportSize({ width: VIEWPORT_WIDTH, height: 844 });
	await loginAsOwner(page);

	await page.goto('/prompts');
	await page.getByRole('button', { name: 'New prompt' }).first().click();
	await expect(page.getByRole('heading', { name: 'New prompt' })).toBeVisible({ timeout: 10000 });

	const uniqueName = `E2E phone prompt ${Date.now()}`;
	await page.getByPlaceholder('Content preview is used when unnamed').fill(uniqueName);
	const segmentEditor = page.locator('.inline-chip-editor[role="textbox"]').first();
	await expect(segmentEditor).toBeVisible({ timeout: 10000 });
	await segmentEditor.click();
	await page.keyboard.type(
		'portrait of a woman, 85mm lens, shallow depth of field, soft window light, extremelylongunbrokenwordthatwouldneverfitinsideanarrowcardwithoutclipping'.repeat(2)
	);

	const created = page.waitForResponse(
		(r) => r.url().includes('/api/prompts') && r.request().method() === 'POST',
		{ timeout: 15000 }
	);
	await page.getByRole('button', { name: 'Create', exact: true }).click();
	expect((await created).ok()).toBeTruthy();
	await expect(page.getByText('Prompt created')).toBeVisible({ timeout: 5000 });

	await page.getByRole('button', { name: 'Back to Prompts' }).click();
	const cards = page.locator('[data-prompt-card]');
	await expect(cards.first()).toBeVisible({ timeout: 10000 });

	await screenshot(page, JOURNEY, 'phone-list');

	const count = await cards.count();
	expect(count).toBeGreaterThan(0);
	for (let index = 0; index < count; index += 1) {
		const card = cards.nth(index);
		const box = await card.boundingBox();
		expect(box, `card ${index} has a box`).toBeTruthy();
		expect(box!.x, `card ${index} left edge`).toBeGreaterThanOrEqual(GUTTER - 1);
		expect(box!.x + box!.width, `card ${index} right edge`).toBeLessThanOrEqual(VIEWPORT_WIDTH - GUTTER + 1);

		const meta = card.getByText(/never used|used \d+×/).first();
		await expect(meta).toBeVisible();
		const metaBox = await meta.boundingBox();
		expect(metaBox, `card ${index} meta has a box`).toBeTruthy();
		expect(metaBox!.x + metaBox!.width, `card ${index} meta right edge`).toBeLessThanOrEqual(VIEWPORT_WIDTH - GUTTER + 1);
	}
});
