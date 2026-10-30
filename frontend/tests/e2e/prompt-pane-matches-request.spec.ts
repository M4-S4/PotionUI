import { test, expect, type Page, type Locator } from '@playwright/test';
import { loginAsOwner, scopedStorageKey } from './helpers';
import { installAndSelectImagePreset } from './presetPreamble';
import { makeModelFieldsOptional } from './generationMocks';

const LEGACY_PROMPT = 'legacy plain prompt, 1girl, solo, sitting';

test.afterEach(async ({ page }) => {
	await page.unrouteAll({ behavior: 'ignoreErrors' });
});

function positiveList(page: Page): Locator {
	return page.locator('div[role="list"][aria-label="Positive segments"]');
}

test('a prompt stored only as a plain string is shown in the editor and is what Generate sends', async ({ page }) => {
	test.setTimeout(180000);
	await makeModelFieldsOptional(page);
	await loginAsOwner(page);
	if (!(await installAndSelectImagePreset(page))) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return;
	}
	await page.setViewportSize({ width: 1440, height: 960 });
	await expect(positiveList(page)).toBeVisible({ timeout: 20000 });

	const key = await scopedStorageKey(page, 'potionui_tabs_state');
	await expect
		.poll(() => page.evaluate((storageKey) => localStorage.getItem(storageKey), key), { timeout: 10000 })
		.not.toBeNull();
	const stored = JSON.parse((await page.evaluate((storageKey) => localStorage.getItem(storageKey), key)) as string);
	for (const tab of stored.tabs) {
		tab.prompt = LEGACY_PROMPT;
		tab.promptSegments = [];
		tab.promptTabs = undefined;
	}
	await page.addInitScript(
		({ storageKey, value }) => localStorage.setItem(storageKey, value),
		{ storageKey: key, value: JSON.stringify(stored) }
	);
	await page.reload();
	await page.waitForLoadState('networkidle');
	await expect(positiveList(page)).toBeVisible({ timeout: 20000 });
	await expect(positiveList(page)).toContainText(LEGACY_PROMPT);

	await page.route('**/api/generations/start', (route) =>
		route.fulfill({ status: 400, json: { success: false, error: 'captured_by_spec', message: 'captured by spec' } })
	);
	const request = page.waitForRequest(
		(candidate) => candidate.method() === 'POST' && candidate.url().endsWith('/api/generations/start')
	);
	await page.getByRole('button', { name: /^Generate/ }).first().click();
	const sent = JSON.parse((await request).postData() ?? '{}');

	expect(sent.prompts[0].positive).toBe(LEGACY_PROMPT);
	expect(sent.prompt_state.promptSegments[0].content).toBe(LEGACY_PROMPT);
});

test('clearing the editor leaves Generate blocked on a missing prompt even though the tab once held one', async ({ page }) => {
	test.setTimeout(180000);
	await makeModelFieldsOptional(page);
	await loginAsOwner(page);
	if (!(await installAndSelectImagePreset(page))) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return;
	}
	await page.setViewportSize({ width: 1440, height: 960 });
	await expect(positiveList(page)).toBeVisible({ timeout: 20000 });

	const starts: string[] = [];
	await page.route('**/api/generations/start', (route) => {
		starts.push(route.request().url());
		return route.fulfill({ status: 400, json: { success: false, error: 'captured_by_spec', message: 'captured by spec' } });
	});

	const editor = positiveList(page).locator('.inline-chip-editor[role="textbox"]').first();
	await editor.click();
	await page.keyboard.type('a prompt that will be erased');
	await expect(page.getByRole('button', { name: /^Generate/ }).first()).toBeEnabled();
	await page.keyboard.press('Control+A');
	await page.keyboard.press('Backspace');
	await expect(positiveList(page)).not.toContainText('erased');

	const blocked = page.getByRole('button', { name: 'Missing prompt', exact: true });
	await expect(blocked).toBeDisabled();
	await expect(page.getByRole('region', { name: 'Generation status' })).toContainText("Can't generate yet");
	await blocked.click({ force: true });
	expect(starts).toEqual([]);
});
