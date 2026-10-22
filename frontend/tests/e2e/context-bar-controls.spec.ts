import { test, expect, type Locator, type Page } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';
import { installAndSelectImagePreset } from './presetPreamble';

const JOURNEY = 'context-bar-controls';
const TOOL_SELECTOR = '[data-context-tool]:not([aria-pressed="true"]), [data-testid="formulas-button"], [data-testid="context-tools-more"]';

async function setup(page: Page): Promise<boolean> {
	await loginAsOwner(page);
	const preset = await installAndSelectImagePreset(page);
	if (!preset) return false;
	await page.setViewportSize({ width: 1440, height: 960 });
	await expect(page.getByTestId('context-bar')).toBeVisible({ timeout: 20000 });
	await expect(page.getByTestId('session-save')).toBeVisible({ timeout: 15000 });
	return true;
}

async function box(locator: Locator) {
	const found = await locator.boundingBox();
	expect(found).not.toBeNull();
	return found!;
}

async function fillOf(locator: Locator): Promise<string> {
	return locator.evaluate((el) => getComputedStyle(el).backgroundColor);
}

test('Save sits at the same 32px height as the context tools and lines up with them', async ({ page }) => {
	test.setTimeout(120000);
	if (!(await setup(page))) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return;
	}
	const save = await box(page.getByTestId('session-save'));
	expect(save.height).toBeCloseTo(32, 0);

	const tool = page.locator(TOOL_SELECTOR).first();
	if (await tool.count()) {
		const toolBox = await box(tool);
		expect(toolBox.height).toBeCloseTo(32, 0);
		expect(Math.abs(save.y - toolBox.y)).toBeLessThanOrEqual(1);
		expect(Math.abs(save.y + save.height - (toolBox.y + toolBox.height))).toBeLessThanOrEqual(1);
	}
	await screenshot(page, JOURNEY, '01-dark');
});

test('context tools take a white fill and a grey hover on the light canvas and keep the surface fills in dark', async ({ page }) => {
	test.setTimeout(120000);
	if (!(await setup(page))) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return;
	}
	const tool = page.locator(TOOL_SELECTOR).first();
	if (!(await tool.count())) {
		test.skip(true, 'This preset exposes no context tools.');
		return;
	}
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
	await expect.poll(() => fillOf(tool)).toBe('rgb(31, 33, 38)');
	await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
	await expect.poll(() => fillOf(tool)).toBe('rgb(255, 255, 255)');
	await tool.hover();
	await expect.poll(() => fillOf(tool)).toBe('rgb(241, 241, 241)');
	await screenshot(page, JOURNEY, '02-light');
});

test('a modal header is 56px tall on desktop', async ({ page }) => {
	test.setTimeout(120000);
	if (!(await setup(page))) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return;
	}
	await page.locator('button[aria-label="Save as a new session"]').click();
	const dialog = page.getByRole('dialog', { name: 'Save New Session' });
	await expect(dialog).toBeVisible();
	const header = await box(dialog.locator(':scope > div').first());
	expect(header.height).toBeCloseTo(56, 0);
	const close = await box(dialog.getByRole('button', { name: 'Close modal' }));
	expect(close.height).toBeCloseTo(32, 0);
	await screenshot(page, JOURNEY, '03-modal-header');
});
