import { test, expect, type Page } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';
import { chooseLayout, installAndSelectImagePreset } from './presetPreamble';

const JOURNEY = 'prompt-header-overflow';
const LABELS = ['Styles', 'Variables'];

async function expectNothingCutOff(page: Page) {
	const header = page.locator('header.composer-toolbar').first();
	await expect(header).toBeVisible({ timeout: 15000 });
	const box = await header.boundingBox();
	expect(box).not.toBeNull();
	const slots = header.locator('[data-toolbar-action]');
	const count = await slots.count();
	for (let i = 0; i < count; i++) {
		if (!(await slots.nth(i).isVisible())) continue;
		const slot = await slots.nth(i).boundingBox();
		expect(slot).not.toBeNull();
		expect(slot!.x).toBeGreaterThanOrEqual(box!.x - 1);
		expect(slot!.x + slot!.width).toBeLessThanOrEqual(box!.x + box!.width + 1);
	}
	const overflowing = await header.evaluate((el) => el.scrollWidth > el.clientWidth + 1);
	expect(overflowing).toBe(false);
	const truncatedTitles = await header.evaluate((el) =>
		Array.from(el.querySelectorAll<HTMLElement>('[role="tab"] .truncate, .prompt-title')).filter(
			(title) => title.scrollWidth > title.clientWidth + 1
		).length
	);
	expect(truncatedTitles).toBe(0);
}

async function reachable(page: Page, label: string): Promise<boolean> {
	const header = page.locator('header.composer-toolbar').first();
	const slot = header.locator('[data-toolbar-action]').filter({ has: page.getByRole('button', { name: new RegExp(`^${label}`) }) });
	if ((await slot.count()) > 0 && (await slot.first().isVisible())) return true;
	await header.getByRole('button', { name: 'Library' }).click();
	const found = (await page.getByRole('menuitem', { name: new RegExp(`^${label}`) }).count()) > 0;
	await page.keyboard.press('Escape');
	return found;
}

test('prompt header steps down its ladder as the prompts pane is dragged narrow', async ({ page }) => {
	test.setTimeout(90000);
	await loginAsOwner(page);
	await page.setViewportSize({ width: 1600, height: 900 });
	await page.goto('/generate');
	await installAndSelectImagePreset(page);

	await chooseLayout(page, 'Three panes');

	const handle = page.getByTestId('prompts-workbench-handle');
	await expect(handle).toBeVisible({ timeout: 20000 });
	await expectNothingCutOff(page);
	await screenshot(page, JOURNEY, 'wide');

	const header = page.locator('header.composer-toolbar').first();
	await expect(header).toHaveAttribute('data-icons', '');
	const startWidth = (await header.boundingBox())!.width;
	const handleBox = (await handle.boundingBox())!;

	await page.mouse.move(handleBox.x + handleBox.width / 2, handleBox.y + handleBox.height / 2);
	await page.mouse.down();
	await page.mouse.move(handleBox.x - Math.round(startWidth * 0.6), handleBox.y + handleBox.height / 2, { steps: 12 });
	await page.mouse.up();

	await expect.poll(async () => (await header.boundingBox())!.width, { timeout: 5000 }).toBeLessThan(startWidth);
	await expectNothingCutOff(page);

	await header.evaluate((el) => {
		(el as HTMLElement).style.maxWidth = '80px';
	});

	await expect(header).toHaveAttribute('data-icons', /library/, { timeout: 5000 });
	await expect(header).toHaveAttribute('data-folded', /\S/, { timeout: 5000 });
	await expect(header.getByRole('button', { name: 'Library' })).toBeVisible();
	await expectNothingCutOff(page);
	await screenshot(page, JOURNEY, 'narrow');

	for (const label of LABELS) {
		expect(await reachable(page, label), `${label} is visible or in the Library menu`).toBe(true);
	}
});
