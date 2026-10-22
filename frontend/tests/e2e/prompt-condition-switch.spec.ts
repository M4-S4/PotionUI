import { test, expect, type Page, type Locator } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';
import { installAndSelectImagePreset } from './presetPreamble';

const JOURNEY = 'prompt-condition-switch';
const LONG = 'a long hooded cloak woven from silver thread and moonlight'.padEnd(199, ', trimmed with fox fur') + 's';

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

async function segmentEditor(page: Page): Promise<Locator> {
	const editor = page.locator('.inline-chip-editor[role="textbox"]').first();
	await expect(editor).toBeVisible({ timeout: 20000 });
	return editor;
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

async function defineChoice(modal: Locator, name: string, options: string[]) {
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
	await showNameField(modal);
	await modal.getByLabel('Variable name').fill(name);
	await modal.getByRole('tab', { name: 'Number' }).click();
	await expect(modal.getByTestId('number-editor')).toBeVisible();
}

async function closeManager(modal: Locator) {
	await modal.getByRole('button', { name: /^Done/ }).click();
	await expect(modal).toBeHidden();
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

const conditionModal = (page: Page) => page.getByRole('dialog', { name: 'Condition', exact: true });
const branch = (dialog: Locator, index: number) => dialog.getByTestId('condition-branch').nth(index);
const saveButton = (dialog: Locator) => dialog.getByRole('button', { name: /^Save/ });

async function insertCondition(page: Page): Promise<Locator> {
	const editor = await segmentEditor(page);
	await editor.click();
	await page.keyboard.type('portrait, ');
	await page.getByRole('button', { name: /^More actions for/ }).first().click();
	await page.getByRole('menuitem', { name: /Insert a condition/ }).click();
	const dialog = conditionModal(page);
	await expect(dialog).toBeVisible();
	return dialog;
}

async function chooseVariable(page: Page, row: Locator, name: string) {
	await row.getByTestId('variable-select').click();
	await page.locator(`[role="option"][data-variable="${name}"]`).click();
}

async function chooseValue(page: Page, row: Locator, value: string) {
	await row.getByTestId('option-select').click();
	const picker = page.getByTestId('option-picker');
	await picker.getByRole('combobox').fill(value.slice(0, 24));
	await picker.locator('[role="option"]').filter({ hasText: value }).first().click();
	await expect(picker).toBeHidden();
}

test.describe('prompt condition switch', () => {
	test.beforeEach(async ({ page }) => {
		test.setTimeout(180000);
		if (!(await openGeneratePage(page))) {
			test.skip(true, 'No native image preset available on this throwaway instance.');
		}
	});

	test('a choice switch on a long option saves as soon as the else if is filled in', async ({ page }) => {
		await captureStart(page);
		const modal = await openVariableManager(page);
		await defineChoice(modal, 'look', ['linen shirt', LONG, 'velvet coat']);
		await closeManager(modal);

		const dialog = await insertCondition(page);
		await chooseVariable(page, branch(dialog, 0), 'look');
		await chooseValue(page, branch(dialog, 0), 'linen shirt');
		await dialog.getByLabel('If text').fill('z');
		await dialog.getByRole('button', { name: 'Else if' }).click();
		await expect(saveButton(dialog)).toBeEnabled();
		await expect(branch(dialog, 1).getByTestId('option-select')).toContainText(LONG);
		await chooseValue(page, branch(dialog, 1), LONG);
		await expect(branch(dialog, 1).getByTestId('option-select')).toContainText(LONG);
		await dialog.getByLabel('Else if text').fill('o');
		await expect(dialog.getByTestId('condition-invalid')).toHaveCount(0);
		await expect(saveButton(dialog)).toBeEnabled();
		await settleAnimations(page);
		await screenshot(page, JOURNEY, '01-choice-switch-1440');
		await saveButton(dialog).click();
		await expect(dialog).toBeHidden();

		const editor = await segmentEditor(page);
		await expect(editor.locator('.condition-chip .chip-state')).toHaveText('+1');
		const request = await generateAndCapture(page);
		expect(request.prompts[0].positive).toContain(`<<if $look = "linen shirt">>z<<else if $look = "${LONG}">>o<<end>>`);
	});

	test('a number switch keeps the typed value without leaving the field', async ({ page }) => {
		await captureStart(page);
		const modal = await openVariableManager(page);
		await defineNumber(modal, 'age');
		await closeManager(modal);

		const dialog = await insertCondition(page);
		await chooseVariable(page, branch(dialog, 0), 'age');
		await branch(dialog, 0).getByLabel('Operator').selectOption('=');
		await branch(dialog, 0).getByLabel('Value', { exact: true }).fill('20');
		await dialog.getByLabel('If text').fill('young');
		await dialog.getByRole('button', { name: 'Else if' }).click();
		await branch(dialog, 1).getByLabel('Operator').selectOption('=');
		await dialog.getByLabel('Else if text').fill('grown');
		await branch(dialog, 1).getByLabel('Value', { exact: true }).fill('40');
		await page.keyboard.press('Control+Enter');
		await expect(dialog).toBeHidden();

		const request = await generateAndCapture(page);
		expect(request.prompts[0].positive).toContain('<<if $age = 20>>young<<else if $age = 40>>grown<<end>>');
	});

	test('says which branch is unfinished and turns a nested condition into an else if', async ({ page }) => {
		const modal = await openVariableManager(page);
		await defineNumber(modal, 'age');
		await closeManager(modal);

		const dialog = await insertCondition(page);
		await chooseVariable(page, branch(dialog, 0), 'age');
		await branch(dialog, 0).getByLabel('Operator').selectOption('in');
		await branch(dialog, 0).getByLabel('From', { exact: true }).fill('90');
		await branch(dialog, 0).getByLabel('To', { exact: true }).fill('10');
		await expect(branch(dialog, 0).getByTestId('condition-problem')).toHaveText(['The From number for $age is above To.']);
		await expect(branch(dialog, 0)).toHaveAttribute('data-invalid', 'true');
		await expect(saveButton(dialog)).toBeDisabled();
		await branch(dialog, 0).getByLabel('To', { exact: true }).fill('95');
		await expect(saveButton(dialog)).toBeEnabled();

		await dialog.getByRole('button', { name: 'Otherwise' }).click();
		await dialog.getByLabel('Otherwise text').fill('<<if $age < 30>>young<<end>>');
		await expect(dialog.getByTestId('condition-problem')).toContainText([
			"This text contains another condition. Conditions can't be nested, so add an Else if instead."
		]);
		await expect(saveButton(dialog)).toBeDisabled();
		await settleAnimations(page);
		await screenshot(page, JOURNEY, '02-nested-problem-1440');
		await dialog.getByRole('button', { name: 'Use Else if' }).click();
		await expect(dialog.getByTestId('condition-branch')).toHaveCount(2);
		await expect(branch(dialog, 1)).toHaveAttribute('data-kind', 'elseif');
		await expect(saveButton(dialog)).toBeEnabled();
	});
});
