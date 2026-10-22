import { test, expect, type Page, type Locator } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';
import { installAndSelectImagePreset } from './presetPreamble';
import { makeModelFieldsOptional } from './generationMocks';

const JOURNEY = 'segment-variables';

async function openGeneratePage(page: Page): Promise<boolean> {
	await loginAsOwner(page);
	await makeModelFieldsOptional(page);
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
	const button = page.getByRole('button', { name: /^Variables/ });
	try {
		await button.waitFor({ state: 'visible', timeout: 10000 });
	} catch {
		const modeButton = page.getByRole('button', { name: /txt2img|text to image/i }).first();
		if ((await modeButton.count()) > 0) await modeButton.click();
		await button.waitFor({ state: 'visible', timeout: 10000 });
	}
	await button.click();
	const modal = page.getByRole('dialog', { name: 'Prompt variables', exact: true });
	await expect(modal).toBeVisible({ timeout: 10000 });
	return modal;
}

async function openVariableManagerResponsive(page: Page): Promise<Locator> {
	const sheet = page.getByRole('dialog', { name: 'Prompt', exact: true });
	const pill = page.getByRole('button', { name: 'Prompt', exact: true });
	if ((await pill.count()) > 0 && (await pill.first().isVisible())) {
		await pill.first().click();
		await expect(sheet).toBeVisible();
	}
	await page.getByRole('button', { name: /^Variables/ }).first().click();
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

async function defineChoice(modal: Locator, name: string, options: string[], first: boolean) {
	if (!first) await modal.getByRole('button', { name: 'Add variable' }).click();
	await modal.getByLabel('Variable name').fill(name);
	await modal.getByRole('tab', { name: 'Choice' }).click();
	const fields = modal.locator('textarea');
	while ((await fields.count()) < options.length) {
		await modal.getByRole('button', { name: 'Add option' }).click();
	}
	for (let i = 0; i < options.length; i++) {
		await modal.getByLabel(`Option ${i + 1}`, { exact: true }).fill(options[i]);
	}
}

async function closeManager(page: Page, modal: Locator) {
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

async function typeTwoUsages(page: Page, name: string) {
	const editor = await segmentEditor(page);
	await editor.click();
	await page.keyboard.type(`a ${'$'}{${name}} and ${'$'}{${name}} end`);
	await expect(editor.locator('.variable-chip')).toHaveCount(2);
	return editor;
}

async function openChipPopover(page: Page, editor: Locator, index: number) {
	await editor.locator('.variable-chip .chip-main').nth(index).dispatchEvent('mousedown');
	const popover = page.locator('.variable-popover').first();
	await expect(popover).toBeVisible();
	return popover;
}

const LONG_OPTIONS = Array.from(
	{ length: 12 },
	(_, i) =>
		`Option ${i + 1}: a deliberately long description of a scene with weathered stone, drifting fog, low amber light and a distant figure on the ridge line number ${i + 1}`
);

test.describe('segment variables', () => {
	test.beforeEach(async ({ page }) => {
		test.setTimeout(180000);
		if (!(await openGeneratePage(page))) {
			test.skip(true, 'No native image preset available on this throwaway instance.');
		}
	});

	test('independent usages show n/N and send name and name-2', async ({ page }) => {
		await captureStart(page);
		const modal = await openVariableManager(page);
		await defineChoice(modal, 'mood', ['calm', 'tense', 'wild'], true);
		await closeManager(page, modal);

		const editor = await typeTwoUsages(page, 'mood');
		await expect(editor.locator('.variable-chip .chip-state')).toHaveText(['1/2', '2/2']);

		const sent = await generateAndCapture(page);
		const positive: string = sent.prompts[0].positive;
		expect(positive).toContain('${mood}');
		expect(positive).toContain('${mood-2}');
		expect(Object.keys(sent.variables)).toEqual(expect.arrayContaining(['mood', 'mood-2']));
		await screenshot(page, JOURNEY, 'independent-usages');
	});

	test('manager shows a used tick once the variable is typed into the prompt', async ({ page }) => {
		const modal = await openVariableManager(page);
		await defineChoice(modal, 'mood', ['calm', 'tense', 'wild'], true);
		await expect(modal.getByTestId('variable-used-tick')).toHaveCount(0);
		await closeManager(page, modal);

		await typeTwoUsages(page, 'mood');

		for (const [width, height] of [
			[1440, 960],
			[390, 844]
		]) {
			await page.setViewportSize({ width, height });
			const reopened = await openVariableManagerResponsive(page);
			const tick = reopened.getByTestId('variable-used-tick');
			await expect(tick).toHaveCount(1);
			await expect(tick).toHaveAttribute('aria-label', /Used in the prompt 2 times/);
			await settleAnimations(page);
			if (width >= 1024) {
				await tick.hover();
				await expect(page.getByText('Used in the prompt · 2')).toBeVisible();
			}
			await screenshot(page, JOURNEY, `used-tick-${width}`);
			await closeManager(page, reopened);
			const promptSheet = page.getByRole('dialog', { name: 'Prompt', exact: true });
			if (await promptSheet.isVisible()) {
				await promptSheet.getByRole('button', { name: 'Done' }).click();
				await expect(promptSheet).toBeHidden();
			}
		}
	});

	test('Linked sends a single shared name', async ({ page }) => {
		await captureStart(page);
		const modal = await openVariableManager(page);
		await defineChoice(modal, 'mood', ['calm', 'tense', 'wild'], true);
		await closeManager(page, modal);

		const editor = await typeTwoUsages(page, 'mood');
		const popover = await openChipPopover(page, editor, 0);
		await popover.getByRole('button', { name: 'Linked', exact: true }).click();
		await expect(editor.locator('.variable-chip .chip-state')).toHaveText(['Linked 2', 'Linked 2']);
		const stillOpen = page.locator('.variable-popover');
		await expect(stillOpen).toBeVisible();
		await expect(stillOpen.getByRole('button', { name: 'Linked', exact: true })).toHaveClass(/active/);
		await page.keyboard.press('Escape');

		const sent = await generateAndCapture(page);
		const positive: string = sent.prompts[0].positive;
		expect(positive).toContain('${mood}');
		expect(positive).not.toContain('${mood-2}');
		expect(Object.keys(sent.variables)).not.toContain('mood-2');
	});

	test('long values stay compact and pin mode uses radio rows', async ({ page }) => {
		const modal = await openVariableManager(page);
		await defineChoice(modal, 'scene', LONG_OPTIONS, true);
		await closeManager(page, modal);

		const editor = await segmentEditor(page);
		await editor.click();
		await page.keyboard.type('${scene}');
		await expect(editor.locator('.variable-chip')).toHaveCount(1);
		const popover = await openChipPopover(page, editor, 0);

		const list = popover.locator('.option-list');
		await expect(list).toBeVisible();
		const box = await list.boundingBox();
		expect(box!.height).toBeLessThanOrEqual(260);
		const fontSize = await popover.locator('.option-text').first().evaluate((el) => getComputedStyle(el).fontSize);
		expect(fontSize).toBe('13px');
		await screenshot(page, JOURNEY, 'compact-popover');

		await popover.getByRole('button', { name: 'Pin', exact: true }).click();
		await expect(editor.locator('.variable-chip .chip-state')).toHaveText('Pinned');
		const pinned = page.locator('.variable-popover');
		await expect(pinned).toBeVisible();
		await expect(pinned.locator('select')).toHaveCount(0);
		const radios = pinned.locator('.option-row[role="radio"]');
		await expect(radios).toHaveCount(LONG_OPTIONS.length);
		await radios.nth(4).click();
		await expect(pinned).toBeVisible();
		await expect(pinned.locator('.option-row.current')).toHaveCount(1);
		await expect(radios.nth(4)).toHaveAttribute('aria-checked', 'true');
	});

	test('manager lists many variables as master-detail', async ({ page }) => {
		const modal = await openVariableManager(page);
		for (let i = 0; i < 25; i++) {
			await defineChoice(modal, `var_${String(i).padStart(2, '0')}`, [`first ${i}`, `second ${i}`], i === 0);
		}
		const rows = modal.locator('[role="option"]');
		await expect(rows).toHaveCount(25);
		await screenshot(page, JOURNEY, 'manager-25');

		await rows.nth(3).click();
		await expect(modal.getByLabel('Variable name')).toHaveValue('var_03');
		await expect(modal.getByLabel('Option 1', { exact: true })).toHaveValue('first 3');

		await modal.getByLabel('Filter variables').fill('var_1');
		await expect(rows).toHaveCount(10);
	});
});
