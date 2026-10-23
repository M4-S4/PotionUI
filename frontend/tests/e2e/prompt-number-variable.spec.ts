import { test, expect, type Page, type Locator } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';
import { installAndSelectImagePreset } from './presetPreamble';
import { makeModelFieldsOptional } from './generationMocks';

const JOURNEY = 'prompt-number-variable';

async function openGeneratePage(page: Page): Promise<boolean> {
	await loginAsOwner(page);
	await makeModelFieldsOptional(page);
	await page.emulateMedia({ colorScheme: 'dark' });
	const preset = await installAndSelectImagePreset(page);
	if (!preset) return false;
	await page.setViewportSize({ width: 1440, height: 960 });
	await segmentEditor(page);
	return true;
}

async function segmentEditor(page: Page): Promise<Locator> {
	const editor = page.locator('.inline-chip-editor[role="textbox"]').first();
	await expect(editor).toBeVisible({ timeout: 20000 });
	return editor;
}

async function openVariableManager(page: Page): Promise<Locator> {
	const sheet = page.getByRole('dialog', { name: 'Prompt', exact: true });
	const pill = page.getByRole('button', { name: 'Prompt', exact: true });
	if ((await pill.count()) > 0 && (await pill.first().isVisible())) {
		await pill.first().click();
		await expect(sheet).toBeVisible();
		await page.locator('[data-toolbar-action="variables"] button').first().click();
	} else {
		await page.getByRole('button', { name: 'Variables' }).click();
	}
	const modal = page.getByRole('dialog', { name: 'Prompt variables', exact: true });
	await expect(modal).toBeVisible({ timeout: 10000 });
	return modal;
}

async function settleAnimations(page: Page) {
	await page.evaluate(() =>
		Promise.all(
			document
				.getAnimations()
				.filter((animation) => animation.effect?.getComputedTiming().iterations !== Infinity)
				.map((animation) => animation.finished.catch(() => undefined))
		)
	);
}

async function defineNumber(modal: Locator, name: string) {
	const nameField = modal.getByLabel('Variable name');
	if (!(await nameField.isVisible())) await modal.getByRole('option').first().click();
	await modal.getByLabel('Variable name').fill(name);
	await modal.getByRole('tab', { name: 'Number' }).click();
	await expect(modal.getByTestId('number-editor')).toBeVisible();
}

async function setNumberField(modal: Locator, label: string, value: string) {
	const field = modal.getByLabel(label, { exact: true });
	await field.fill(value);
	await field.press('Enter');
}

async function captureStart(page: Page) {
	await page.route('**/api/generations/start', (route) =>
		route.fulfill({ status: 400, json: { success: false, error: 'captured_by_spec', message: 'captured by spec' } })
	);
}

async function generateAndCapture(page: Page) {
	const request = page.waitForRequest(
		(candidate) => candidate.method() === 'POST' && candidate.url().endsWith('/api/generations/start')
	);
	await page.getByRole('button', { name: /^Generate/ }).first().click();
	return JSON.parse((await request).postData() ?? '{}');
}

test.describe('number variable', () => {
	test.beforeEach(async ({ page }) => {
		test.setTimeout(180000);
		if (!(await openGeneratePage(page))) {
			test.skip(true, 'No native image preset available on this throwaway instance.');
		}
	});

	test('creates a number variable, switches modes and shows the chip', async ({ page }) => {
		await captureStart(page);
		const editor = await segmentEditor(page);
		await editor.click();
		await page.keyboard.type('portrait, ${age} years old');
		await expect(editor.locator('.variable-chip')).toHaveCount(1);

		const modal = await openVariableManager(page);
		await defineNumber(modal, 'age');
		await modal.getByRole('radio', { name: 'Shuffle' }).click();
		await setNumberField(modal, 'Minimum', '18');
		await setNumberField(modal, 'Maximum', '80');
		await expect(modal.getByTestId('number-preview-note')).toContainText('from 63 values');
		await settleAnimations(page);
		await screenshot(page, JOURNEY, '01-shuffle-1440');

		await modal.getByRole('radio', { name: 'Per image' }).click();
		await expect(modal.getByTestId('number-dot')).toHaveCount(4);
		await expect(modal.getByTestId('number-preview-note')).toContainText('63 possible values');
		await settleAnimations(page);
		await screenshot(page, JOURNEY, '02-per-image-1440');

		await modal.getByRole('radio', { name: 'Pin' }).click();
		await setNumberField(modal, 'Pinned value', '47');
		await expect(modal.getByTestId('number-dot')).toHaveCount(1);
		await settleAnimations(page);
		await screenshot(page, JOURNEY, '03-pin-1440');

		await modal.getByRole('button', { name: /^Done/ }).click();
		await expect(modal).toBeHidden();
		await expect(editor.locator('.variable-chip').first()).toContainText('47');

		const pinned = await generateAndCapture(page);
		expect(pinned.variables.age).toBe('47');
	});

	test('per image sends the enumerated group and decimals keep their places', async ({ page }) => {
		await captureStart(page);
		const editor = await segmentEditor(page);
		await editor.click();
		await page.keyboard.type('(red dress:${weight})');
		await expect(editor.locator('.variable-chip')).toHaveCount(1);

		const modal = await openVariableManager(page);
		await defineNumber(modal, 'weight');
		await modal.getByRole('radio', { name: 'Per image' }).click();
		await modal.getByRole('radio', { name: /^Decimal/ }).click();
		await setNumberField(modal, 'Minimum', '0.8');
		await setNumberField(modal, 'Maximum', '1');
		await expect(modal.getByTestId('number-preview-note')).toContainText('possible values');
		await modal.getByRole('button', { name: /^Done/ }).click();
		await expect(editor.locator('.variable-chip').first()).toContainText('per image');

		const request = await generateAndCapture(page);
		expect(request.variables.weight).toBe('{0.8|0.9|1.0}');
	});

	test('warns when the range has more than 200 values', async ({ page }) => {
		const modal = await openVariableManager(page);
		await defineNumber(modal, 'big');
		await setNumberField(modal, 'Maximum', '1000');
		await expect(modal.getByTestId('number-grid-warning')).toContainText('Raise the step');
		await expect(modal.getByTestId('number-track')).toHaveCount(0);
	});

	test('number sheet fits a phone', async ({ page }) => {
		await page.setViewportSize({ width: 390, height: 844 });
		await expect(page.getByRole('button', { name: 'Prompt', exact: true })).toBeVisible();
		const modal = await openVariableManager(page);
		await defineNumber(modal, 'age');
		await modal.getByRole('radio', { name: 'Per image' }).click();
		await modal.getByRole('radio', { name: /^Decimal/ }).click();
		const decimal = modal.getByRole('radio', { name: /^Decimal/ });
		const box = await decimal.boundingBox();
		const group = await modal.getByRole('radiogroup', { name: 'Whole or decimal numbers' }).boundingBox();
		expect(box && group && box.x + box.width <= group.x + group.width + 1).toBeTruthy();
		const scroll = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
		expect(scroll).toBe(true);
		const tops = await Promise.all(
			['Copy', 'Paste…', 'Roll preview'].map(async (label) => {
				const box = await modal.getByRole('button', { name: label, exact: true }).boundingBox();
				return Math.round(box?.y ?? -1);
			})
		);
		expect(new Set(tops).size).toBe(1);
		expect(tops[0]).toBeGreaterThan(0);
		const done = await modal.getByRole('button', { name: /^Done/ }).boundingBox();
		const copy = await modal.getByRole('button', { name: 'Copy', exact: true }).boundingBox();
		expect(done && copy && done.y >= copy.y + copy.height - 1).toBeTruthy();
		await settleAnimations(page);
		await screenshot(page, JOURNEY, '04-number-sheet-390');
	});
});
