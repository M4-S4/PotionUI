import { test, expect, type Page, type Locator } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';
import { installAndSelectImagePreset } from './presetPreamble';

const JOURNEY = 'chip-dropdowns';
const CHIP_POPOVER_WIDTH = 420;
const LONG_OPTIONS = [
	'a fitted phase coat with emitters',
	'a long hooded cloak of silver thread',
	'plain linen work shirt',
	'ceremonial bronze scale armour',
	'a weathered leather duster coat'
];

const VIEWPORTS = [
	{ label: '1440', width: 1440, height: 960 },
	{ label: '390', width: 390, height: 844 }
];

const PHRASE_VALUES = [
	'soft window light',
	'dramatic rim light',
	'golden hour',
	'neon glow',
	'overcast diffusion',
	'candlelight',
	'studio strobe',
	'moonlit haze'
].map((label, index) => ({
	id: `val-${index}`,
	category_id: 'cat-lighting',
	label,
	value: label,
	sort_order: index,
	is_active: true,
	created_at: '2026-01-01T00:00:00Z',
	updated_at: '2026-01-01T00:00:00Z',
	category_path: 'style.lighting'
}));

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

async function mockPhrasebook(page: Page) {
	await page.route('**/api/phrasebook/search**', (route) =>
		route.fulfill({
			json: {
				success: true,
				data: {
					current_category: {
						id: 'cat-lighting',
						name: 'lighting',
						path: 'style.lighting',
						description: '',
						is_active: true,
						created_at: '2026-01-01T00:00:00Z',
						updated_at: '2026-01-01T00:00:00Z'
					},
					child_categories: [],
					values: PHRASE_VALUES,
					path: 'style.lighting',
					total_children: 0,
					total_values: PHRASE_VALUES.length
				}
			}
		})
	);
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

async function showNameField(modal: Locator) {
	const nameField = modal.getByLabel('Variable name');
	if (!(await nameField.isVisible())) await modal.getByRole('option').last().click();
	await expect(nameField).toBeVisible();
}

async function defineChoice(modal: Locator, name: string, options: string[], first: boolean) {
	if (!first) await modal.getByRole('button', { name: 'Add variable' }).click();
	await showNameField(modal);
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

async function defineNumber(modal: Locator, name: string) {
	await modal.getByRole('button', { name: 'Add variable' }).click();
	await showNameField(modal);
	await modal.getByLabel('Variable name').fill(name);
	await modal.getByRole('tab', { name: 'Number' }).click();
	await expect(modal.getByTestId('number-editor')).toBeVisible();
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

async function revealEditor(page: Page): Promise<Locator> {
	const pill = page.getByRole('button', { name: 'Prompt', exact: true });
	const sheet = page.getByRole('dialog', { name: 'Prompt', exact: true });
	if ((await pill.count()) > 0 && (await pill.first().isVisible()) && !(await sheet.isVisible())) {
		await pill.first().click();
		await expect(sheet).toBeVisible();
	}
	return segmentEditor(page);
}

async function expectInsideViewport(locator: Locator, width: number) {
	const box = await locator.boundingBox();
	expect(box).not.toBeNull();
	expect(box!.x).toBeGreaterThanOrEqual(0);
	expect(box!.x + box!.width).toBeLessThanOrEqual(width + 1);
}

for (const viewport of VIEWPORTS) {
	test(`chip dropdowns share one width and the variable picker searches at ${viewport.label}`, async ({ page }) => {
		test.setTimeout(240000);
		await loginAsOwner(page);
		await makeModelFieldsOptional(page);
		await mockPhrasebook(page);
		await page.emulateMedia({ colorScheme: 'dark' });
		const preset = await installAndSelectImagePreset(page);
		test.skip(!preset, 'No native image preset available on this throwaway instance.');
		await page.setViewportSize({ width: viewport.width, height: viewport.height });
		const modal = await openVariableManager(page);
		await defineChoice(modal, 'mood', ['calm', 'serene', 'tense', 'wild', 'quiet'], true);
		await defineChoice(modal, 'hair', ['auburn hair', 'grey hair', 'blonde hair'], false);
		await defineChoice(modal, 'outfit', LONG_OPTIONS, false);
		await defineNumber(modal, 'age');
		await modal.getByRole('button', { name: /^Done/ }).click();
		await expect(modal).toBeHidden();

		let editor = await revealEditor(page);
		await editor.click();
		await page.keyboard.type('portrait of a woman, $mo');
		const variablePicker = page.locator('.variable-picker');
		await expect(variablePicker).toBeVisible();
		await expect(variablePicker.locator('.picker-row').first()).toContainText('mood');
		await expect(variablePicker.locator('.picker-row').first()).toContainText('Choice');
		await expect(variablePicker.locator('mark').first()).toHaveText('mo');
		if (viewport.width >= 768) {
			const box = await variablePicker.boundingBox();
			expect(Math.round(box!.width)).toBe(CHIP_POPOVER_WIDTH);
		}
		await expectInsideViewport(variablePicker, viewport.width);
		await settleAnimations(page);
		await screenshot(page, JOURNEY, `01-insert-variable-menu-${viewport.label}`);
		await page.keyboard.press('Enter');
		await expect(editor.locator('.variable-chip')).toHaveCount(1);

		await page.keyboard.type(' ${age}');
		await editor.locator('.variable-chip .chip-main').first().dispatchEvent('mousedown');
		const variableDialog = page.getByRole('dialog', { name: 'Variable mood' });
		await expect(variableDialog).toBeVisible();
		if (viewport.width >= 768) {
			const box = await variableDialog.boundingBox();
			expect(Math.round(box!.width)).toBe(CHIP_POPOVER_WIDTH);
		}
		await expectInsideViewport(variableDialog, viewport.width);
		await settleAnimations(page);
		await screenshot(page, JOURNEY, `02-variable-chip-dropdown-${viewport.label}`);
		await page.keyboard.press('Escape');
		await expect(variableDialog).toBeHidden();

		await page.getByRole('button', { name: /^More actions for/ }).first().click();
		await page.getByRole('menuitem', { name: /Insert a condition/ }).click();
		const conditionDialog = page.getByRole('dialog', { name: 'Condition', exact: true });
		await expect(conditionDialog).toBeVisible();
		const modalBox = await conditionDialog.boundingBox();
		if (viewport.width >= 768) expect(Math.round(modalBox!.width)).toBe(Math.min(1040, viewport.width - 48));
		else expect(Math.round(modalBox!.width)).toBe(viewport.width);
		await conditionDialog.getByTestId('variable-select').first().click();
		const picker = page.getByTestId('variable-picker');
		await expect(picker).toBeVisible();
		const search = picker.getByRole('combobox', { name: 'Variable' });
		await expect(search).toBeFocused();
		await search.fill('ser');
		await expect(picker.locator('[role="option"][data-variable]')).toHaveCount(1);
		await expect(picker.locator('[role="option"][data-variable]').first()).toContainText('mood');
		await expect(picker.locator('mark').first()).toHaveText('ser');
		await expect(conditionDialog).toBeVisible();
		await expectInsideViewport(picker, viewport.width);
		await settleAnimations(page);
		await screenshot(page, JOURNEY, `03-condition-variable-picker-${viewport.label}`);
		await search.press('Enter');
		await expect(picker).toBeHidden();
		await expect(conditionDialog).toBeVisible();
		await expect(conditionDialog.getByTestId('variable-select').first()).toContainText('$mood');
		await expect(conditionDialog.getByTestId('option-select')).toBeVisible();
		await settleAnimations(page);
		await screenshot(page, JOURNEY, `04-condition-modal-${viewport.label}`);

		await conditionDialog.getByTestId('variable-select').first().click();
		await picker.getByRole('combobox', { name: 'Variable' }).fill('outfit');
		await picker.getByRole('combobox', { name: 'Variable' }).press('Enter');
		await conditionDialog.getByTestId('option-select').first().click();
		const values = page.getByTestId('option-picker');
		await expect(values.getByRole('option')).toHaveCount(LONG_OPTIONS.length);
		await expectInsideViewport(values, viewport.width);
		await values.getByRole('option').nth(1).click();
		await expect(values).toBeHidden();
		await expect(conditionDialog.getByTestId('option-select').first()).toContainText(LONG_OPTIONS[1]);
		await conditionDialog.getByRole('button', { name: 'Else if' }).click();
		await conditionDialog.getByRole('button', { name: 'Otherwise' }).click();
		const overflowing = await conditionDialog.getByTestId('condition-editor-body').evaluate((body) => {
			const bounds = body.getBoundingClientRect();
			return Array.from(body.querySelectorAll('*'))
				.filter((el) => {
					const rect = el.getBoundingClientRect();
					return rect.width > 0 && (rect.right > bounds.right + 1 || rect.left < bounds.left - 1);
				})
				.map((el) => el.tagName + '.' + el.className);
		});
		expect(overflowing).toEqual([]);
		const buttonBoxes = await Promise.all(
			[/^Remove condition/, /^Cancel/, /^Save/].map(async (name) => conditionDialog.getByRole('button', { name }).boundingBox())
		);
		const [removeBox, cancelBox, saveBox] = buttonBoxes;
		expect(removeBox && cancelBox && saveBox).toBeTruthy();
		const overlaps = (a: NonNullable<typeof removeBox>, b: NonNullable<typeof removeBox>) =>
			a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
		expect(overlaps(removeBox!, cancelBox!)).toBe(false);
		expect(overlaps(cancelBox!, saveBox!)).toBe(false);
		expect(overlaps(removeBox!, saveBox!)).toBe(false);
		await settleAnimations(page);
		await screenshot(page, JOURNEY, `05-condition-modal-long-values-${viewport.label}`);
		await conditionDialog.getByRole('button', { name: /^Save/ }).click();
		await expect(conditionDialog).toBeHidden();

		await page.keyboard.press('Control+End');
		await page.keyboard.type(' #style.lighting');
		const phrasePicker = page.locator('.phrasebook-picker');
		await expect(phrasePicker).toBeVisible();
		await expect(phrasePicker.locator('.phrase-value')).toHaveCount(PHRASE_VALUES.length);
		if (viewport.width >= 768) {
			const box = await phrasePicker.boundingBox();
			expect(Math.round(box!.width)).toBe(CHIP_POPOVER_WIDTH);
		}
		await expectInsideViewport(phrasePicker, viewport.width);
		await settleAnimations(page);
		await screenshot(page, JOURNEY, `06-phrasebook-dropdown-${viewport.label}`);
		await phrasePicker.locator('.phrase-value').first().click();
		editor = await segmentEditor(page);
		await expect(editor.locator('.phrase-chip')).toHaveCount(1);
		await editor.locator('.phrase-chip .chip-main').first().dispatchEvent('mousedown');
		const switcher = page.getByRole('dialog').filter({ hasText: 'Change value' });
		await expect(switcher).toBeVisible();
		await settleAnimations(page);
		await screenshot(page, JOURNEY, `07-phrasebook-chip-switcher-${viewport.label}`);
		await page.keyboard.press('Escape');
	});
}
