import { test, expect, type Page, type Locator } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';
import { installAndSelectImagePreset } from './presetPreamble';

const JOURNEY = 'prompt-conditions';

async function makeModelFieldsOptional(page: Page) {
	const relax = (node: unknown): void => {
		if (Array.isArray(node)) return node.forEach(relax);
		if (node && typeof node === 'object') {
			const record = node as Record<string, unknown>;
			if (record.type === 'model' && 'required' in record) record.required = false;
			Object.values(record).forEach(relax);
		}
	};
	await page.route(/\/api\/presets\/[^/]+\/form(\?.*)?$/, async (route) => {
		const response = await route.fetch();
		const body = await response.json();
		relax(body);
		await route.fulfill({ response, json: body });
	});
}

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

async function closeManager(modal: Locator) {
	await modal.getByRole('button', { name: /^Done/ }).click();
	await expect(modal).toBeHidden();
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


const popover = (page: Page) => page.getByRole('dialog', { name: 'Condition', exact: true });

async function insertConditionFromMenu(page: Page) {
	await page.getByRole('button', { name: /^More actions for/ }).first().click();
	await page.getByRole('menuitem', { name: /Insert a condition/ }).click();
}

test.describe('prompt conditions', () => {
	test.beforeEach(async ({ page }) => {
		test.setTimeout(180000);
		if (!(await openGeneratePage(page))) {
			test.skip(true, 'No native image preset available on this throwaway instance.');
		}
	});

	test('inserts a condition from the segment menu, edits it and previews the branch', async ({ page }) => {
		await captureStart(page);
		const modal = await openVariableManager(page);
		await defineNumber(modal, 'age');
		await modal.getByRole('radio', { name: 'Per image' }).click();
		await closeManager(modal);

		const editor = await segmentEditor(page);
		await editor.click();
		await page.keyboard.type('portrait, ');
		await insertConditionFromMenu(page);
		const dialog = popover(page);
		await expect(dialog).toBeVisible();
		await expect(editor.locator('.condition-chip')).toHaveCount(1);
		await settleAnimations(page);
		await screenshot(page, JOURNEY, '01-popover-1440');

		await dialog.getByLabel('Operator').selectOption('>');
		await dialog.getByLabel('If text').fill('grey hair');
		await dialog.getByRole('button', { name: 'Otherwise' }).click();
		await dialog.getByLabel('Otherwise text').fill('auburn hair');
		await expect(dialog.getByTestId('condition-last-run')).toContainText('Decided when you generate');
		await settleAnimations(page);
		await screenshot(page, JOURNEY, '02-popover-otherwise-1440');
		await dialog.getByRole('button', { name: /^Save/ }).click();
		await expect(dialog).toBeHidden();
		await expect(editor.getByTestId('condition-chip-summary')).toContainText('grey hair');
		await expect(editor.locator('.condition-chip .chip-context')).toHaveText('age > 0');
		await screenshot(page, JOURNEY, '03-chip-1440');

		await page.getByRole('button', { name: /^Resolved prompt/ }).click();
		await expect(page.getByTestId('condition-preview-table')).toBeVisible();
		await expect(page.getByTestId('condition-preview-row')).toHaveCount(4);
		await settleAnimations(page);
		await screenshot(page, JOURNEY, '04-preview-1440');

		const request = await generateAndCapture(page);
		expect(request.prompts[0].positive).toContain('<<if $age > 0>>grey hair<<else>>auburn hair<<end>>');
		expect(request.variables.age).toMatch(/^\{0\|1\|/);
	});

	test('inserted grammar becomes a chip and a read variable is sent once', async ({ page }) => {
		await captureStart(page);
		const modal = await openVariableManager(page);
		await defineNumber(modal, 'age');
		await closeManager(modal);

		const editor = await segmentEditor(page);
		await editor.click();
		await page.keyboard.type('a ${age} and ${age}, ');
		await page.keyboard.insertText('<<if $age > 50>>grey<<else>>dark<<end>>');
		await expect(editor.locator('.condition-chip')).toHaveCount(1);
		await expect(editor.locator('.variable-chip .chip-state').first()).toContainText('Linked 2');

		const request = await generateAndCapture(page);
		expect(Object.keys(request.variables)).toEqual(['age']);
	});

	test('flags a condition on a missing variable and still allows generating', async ({ page }) => {
		await captureStart(page);
		const editor = await segmentEditor(page);
		await editor.click();
		await page.keyboard.type('hair ');
		await page.keyboard.insertText('<<if $ghost > 1>>grey<<end>>');
		await page.keyboard.type(' done');
		const chip = editor.locator('.condition-chip');
		await expect(chip).toHaveClass(/broken/);
		await settleAnimations(page);
		await screenshot(page, JOURNEY, '05-broken-chip-1440');
		const request = await generateAndCapture(page);
		expect(request.prompts[0].positive).toContain('<<if $ghost > 1>>grey<<end>>');
	});

	test('typing << opens the condition editor and saving drops the chip where it was typed', async ({ page }) => {
		await captureStart(page);
		const manager = await openVariableManager(page);
		await defineNumber(manager, 'age');
		await closeManager(manager);

		const editor = await segmentEditor(page);
		await editor.click();
		await page.keyboard.type('portrait, soft light');
		for (let step = 0; step < 'soft light'.length; step++) await page.keyboard.press('ArrowLeft');
		await page.keyboard.type('<');
		await expect(popover(page)).toBeHidden();
		await page.keyboard.type('<');
		const dialog = popover(page);
		await expect(dialog).toBeVisible();
		await dialog.getByLabel('Operator').selectOption('>');
		await dialog.getByLabel('If text').fill('grey hair');
		await dialog.getByRole('button', { name: /^Save/ }).click();
		await expect(dialog).toBeHidden();
		await expect(editor.locator('.condition-chip')).toHaveCount(1);
		await expect(editor.getByTestId('condition-chip-summary')).toContainText('grey hair');
		await expect(editor).not.toContainText('<<');

		await page.keyboard.type('!');
		const request = await generateAndCapture(page);
		expect(request.prompts[0].positive).toContain('portrait, <<if $age > 0>>grey hair<<end>>!soft light');
	});

	test('typing << on a later line saves the chip on that line', async ({ page }) => {
		await captureStart(page);
		const manager = await openVariableManager(page);
		await defineNumber(manager, 'age');
		await closeManager(manager);

		const editor = await segmentEditor(page);
		await editor.click();
		await page.keyboard.type('line one');
		await page.keyboard.press('Enter');
		await page.keyboard.type('line two');
		await page.keyboard.press('Enter');
		await page.keyboard.type('line three<<');
		const dialog = popover(page);
		await expect(dialog).toBeVisible();
		await dialog.getByLabel('Operator').selectOption('>');
		await dialog.getByLabel('If text').fill('grey hair');
		await dialog.getByRole('button', { name: /^Save/ }).click();
		await expect(dialog).toBeHidden();
		await expect(editor.locator('.condition-chip')).toHaveCount(1);
		await expect(editor).not.toContainText('<<');

		await page.keyboard.type('!');
		const request = await generateAndCapture(page);
		const positive: string = request.prompts[0].positive;
		expect(positive).toContain('line one\nline two\nline three<<if $age > 0>>grey hair<<end>>!');
	});

	test('cancelling the typed << editor keeps the literal text and a third < does not reopen it', async ({ page }) => {
		await captureStart(page);
		const editor = await segmentEditor(page);
		await editor.click();
		await page.keyboard.type('a cat<<');
		const dialog = popover(page);
		await expect(dialog).toBeVisible();
		await page.keyboard.press('Escape');
		await expect(dialog).toBeHidden();

		await page.keyboard.type('<x');
		await expect(popover(page)).toBeHidden();
		await expect(editor).toContainText('a cat<<<x');
		await expect(editor.locator('.condition-chip')).toHaveCount(0);
	});

	test('the condition modal fits a phone', async ({ page }) => {
		await page.setViewportSize({ width: 390, height: 844 });
		await expect(page.getByRole('button', { name: 'Prompt', exact: true }).first()).toBeVisible();
		const modal = await openVariableManager(page);
		await defineNumber(modal, 'age');
		await closeManager(modal);
		const pill = page.getByRole('button', { name: 'Prompt', exact: true });
		if ((await page.locator('.inline-chip-editor[role="textbox"]').first().isVisible()) === false) {
			await pill.first().click();
		}
		const editor = await segmentEditor(page);
		await editor.click();
		await page.keyboard.type('hair ');
		await page.keyboard.insertText('<<if $age > 50>>grey<<else>>dark<<end>>');
		await expect(editor.locator('.condition-chip')).toHaveCount(1);
		await screenshot(page, JOURNEY, '06-chip-390');
		await editor.locator('.condition-chip .chip-main').first().dispatchEvent('mousedown');
		const dialog = popover(page);
		await expect(dialog).toBeVisible();
		await expect(dialog.getByRole('button', { name: /^Save/ })).toBeInViewport();
		await expect(dialog.getByRole('button', { name: 'Remove condition' })).toBeInViewport();
		const box = await dialog.boundingBox();
		expect(box && box.x >= 0 && box.x + box.width <= 390 + 1).toBeTruthy();
		expect(Math.round(box!.height)).toBeGreaterThan(700);
		const scroll = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
		expect(scroll).toBe(true);
		await settleAnimations(page);
		await screenshot(page, JOURNEY, '07-popover-390');
	});
});
