import { test, expect, type Locator, type Page } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';

const JOURNEY = 'prompt-library-conditions';

const conditionDialog = (page: Page) => page.getByRole('dialog', { name: 'Condition', exact: true });

async function openManager(page: Page, trigger: Locator): Promise<Locator> {
	await trigger.click();
	const modal = page.getByRole('dialog', { name: 'Prompt variables', exact: true });
	await expect(modal).toBeVisible({ timeout: 10000 });
	return modal;
}

async function defineChoice(modal: Locator, name: string, options: string[]) {
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

async function closeManager(modal: Locator) {
	await modal.getByRole('button', { name: /^Done/ }).click();
	await expect(modal).toBeHidden();
}

test('segment detail uses composer chips and composes a condition from the active tab variables', async ({ page }) => {
	test.setTimeout(90000);
	await loginAsOwner(page);

	await page.goto('/prompts?section=segments&new=1');
	await expect(page.getByRole('heading', { name: 'New segment' })).toBeVisible({ timeout: 15000 });

	const editor = page.locator('.segment-composer .segment-content .inline-chip-editor[role="textbox"]');
	await expect(editor).toBeVisible({ timeout: 10000 });

	const modal = await openManager(page, page.getByRole('button', { name: 'Manage variables' }));
	await defineChoice(modal, 'mood', ['calm', 'tense']);
	await closeManager(modal);
	await expect(page.getByTestId('variables-list')).toContainText('$mood');

	await editor.click();
	await page.keyboard.type('portrait, <<');
	const dialog = conditionDialog(page);
	await expect(dialog).toBeVisible({ timeout: 10000 });
	await expect(dialog).toContainText('$mood');
	await dialog.getByRole('button', { name: /^Save/ }).click();
	await expect(dialog).toBeHidden();

	await expect(editor.locator('.condition-chip')).toHaveCount(1);
	await expect(editor.locator('.condition-chip.broken')).toHaveCount(0);
	await expect(editor.locator('.chip')).not.toHaveCount(0);
	await screenshot(page, JOURNEY, 'segment-condition');

	console.log(`[${JOURNEY}] segment detail composed a condition against the active tab variables`);
});

test('template detail offers the variables button and opens the condition editor in a slot', async ({ page }) => {
	test.setTimeout(90000);
	await loginAsOwner(page);

	await page.goto('/prompts?section=templates&new=1');
	await expect(page.getByRole('heading', { name: 'New template' })).toBeVisible({ timeout: 15000 });

	const editor = page.locator('.inline-chip-editor[role="textbox"]').first();
	await expect(editor).toBeVisible({ timeout: 10000 });

	const modal = await openManager(page, page.getByRole('button', { name: /^Variables/ }).first());
	await defineChoice(modal, 'mood', ['calm', 'tense']);
	await closeManager(modal);
	await expect(page.getByTestId('variables-list')).toContainText('$mood');

	await editor.click();
	await page.keyboard.type('portrait, <<');
	const dialog = conditionDialog(page);
	await expect(dialog).toBeVisible({ timeout: 10000 });
	await expect(dialog).toContainText('$mood');
	await page.keyboard.press('Escape');
	await expect(dialog).toBeHidden();
	await expect(editor).toContainText('<<');
	await screenshot(page, JOURNEY, 'template-condition-editor');

	console.log(`[${JOURNEY}] template detail opened the condition editor against the active tab variables`);
});
