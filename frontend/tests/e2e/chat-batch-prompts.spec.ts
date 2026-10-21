import { test, expect, type Page } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot } from './helpers';
import { installAndSelectImagePreset } from './presetPreamble';
import { startFakeLLM, seedFakeLlmConfig, type FakeLLMServer } from './fake-llm';

const JOURNEY = 'chat-batch-prompts';
const BACKEND = process.env.E2E_BACKEND_URL || 'http://127.0.0.1:8055';

const NIGHT = 'a quiet village at night, lit windows';

let fake: FakeLLMServer;

test.beforeEach(async () => {
	fake = await startFakeLLM();
});

test.afterEach(async ({ page }) => {
	await fake.close();
	await page.unrouteAll({ behavior: 'ignoreErrors' });
});

async function enableBatch(page: Page) {
	await page.route(/\/api\/presets(\?[^/]*)?$/, async (route) => {
		if (route.request().method() !== 'GET') return route.fallback();
		const response = await route.fetch();
		const body = await response.json();
		if (Array.isArray(body?.data)) {
			body.data = body.data.map((preset: Record<string, unknown>) => ({ ...preset, supports_prompt_batch: true }));
		}
		await route.fulfill({ response, json: body });
	});
}

async function send(page: Page, text: string) {
	const composer = page.locator('.composer:has(button[aria-label="Send message"])');
	const chipInput = composer.locator('[role="textbox"][aria-placeholder]');
	await expect(chipInput).toHaveAttribute('contenteditable', 'true');
	await chipInput.click();
	await page.keyboard.type(text);
	await composer.locator('button[aria-label="Send message"]').click();
}

function applyButtons(page: Page) {
	return page.locator('.assistant-message').last().locator('[data-testid="segment-action-apply"]');
}

async function openChat(page: Page) {
	const phone = page.viewportSize()!.width < 768;
	await page.getByRole('button', { name: phone ? 'Open AI chat' : 'AI Chat', exact: true }).click();
	await expect(page.locator('.composer:has(button[aria-label="Send message"])').filter({ visible: true })).toHaveCount(1, {
		timeout: 15000
	});
}

test.use({ viewport: { width: 1440, height: 900 } });

test('assistant adds a second prompt, the Prompt header shows 2 prompts, and the add can be undone', async ({ page }) => {
	test.setTimeout(180000);
	await enableBatch(page);
	await loginAsOwner(page);
	const token = await ownerToken(page);
	await seedFakeLlmConfig(page.request, BACKEND, token, fake.url);

	const preset = await installAndSelectImagePreset(page);
	if (!preset) {
		test.skip(true, 'No native image preset available on this throwaway instance to render the prompt editor with.');
		return;
	}
	await expect(page.locator('[data-prompt-picker]')).toHaveText(/1 prompt/, { timeout: 20000 });

	await page.getByRole('button', { name: 'AI Chat' }).click();
	await expect(page.locator('.composer:has(button[aria-label="Send message"])')).toBeVisible({ timeout: 15000 });

	fake.enqueue({
		kind: 'text',
		text:
			'Here is a night version as a second prompt:\n' +
			`<tool_action type="add_prompt" name="Night version">${NIGHT}</tool_action>`
	});
	await send(page, 'Add a second prompt with a night version');

	const reply = page.locator('.assistant-message').last();
	await expect(reply.getByText('Add prompt 2 · Night version')).toBeVisible({ timeout: 30000 });
	await expect(applyButtons(page)).toHaveCount(1);
	await screenshot(page, JOURNEY, 'add-prompt-proposed-1440');

	await applyButtons(page).click();
	await expect(applyButtons(page)).toHaveText(/Added/);
	await expect(page.locator('[data-prompt-picker]')).toHaveAttribute('aria-label', 'Prompt 2 of 2');
	await expect(page.locator('.inline-chip-editor[role="textbox"]').filter({ hasText: NIGHT }).first()).toBeVisible();

	await page.getByRole('button', { name: 'Close chat', exact: true }).click();
	await page.locator('[data-prompt-picker]').click();
	const menu = page.getByRole('dialog', { name: 'Prompts' });
	await expect(menu).toContainText('2 prompts');
	await expect(menu).toContainText('Night version');
	await screenshot(page, JOURNEY, 'two-prompts-1440');
	await page.keyboard.press('Escape');

	await page.setViewportSize({ width: 390, height: 844 });
	await screenshot(page, JOURNEY, 'two-prompts-390');

	await page.setViewportSize({ width: 1440, height: 900 });
	await openChat(page);
	await reply.locator('[data-testid="segment-action-undo"]').click();
	await expect(page.locator('[data-prompt-picker]')).toHaveText(/1 prompt/);
	await expect(applyButtons(page)).toHaveText('Add');
	await screenshot(page, JOURNEY, 'add-undone');

	console.log(
		`[${JOURNEY}] preset ${preset.id}: add_prompt applied (2 prompts) and undone; ` +
			`fake-llm requests=${fake.requests.length}, unconsumed turns=${fake.pending()}`
	);
});

test('prompt actions render at phone width', async ({ page }) => {
	test.setTimeout(180000);
	await enableBatch(page);
	await loginAsOwner(page);
	const token = await ownerToken(page);
	await seedFakeLlmConfig(page.request, BACKEND, token, fake.url);

	const preset = await installAndSelectImagePreset(page);
	if (!preset) {
		test.skip(true, 'No native image preset available on this throwaway instance to render the prompt editor with.');
		return;
	}
	await page.getByRole('button', { name: 'AI Chat' }).click();
	await expect(page.locator('.composer:has(button[aria-label="Send message"])')).toBeVisible({ timeout: 15000 });

	fake.enqueue({
		kind: 'text',
		text:
			'Three changes:\n' +
			`<tool_action type="add_prompt" name="Night version">${NIGHT}</tool_action>\n` +
			'<tool_action type="select_prompt" prompt="0"></tool_action>\n' +
			'<tool_action type="rename_prompt" prompt="0" name="Day version"></tool_action>'
	});
	await send(page, 'Set up day and night prompts');

	const reply = page.locator('.assistant-message').last();
	await expect(reply.getByText('Switch to prompt 1')).toBeVisible({ timeout: 30000 });
	await expect(reply.getByText('Rename prompt 1')).toBeVisible();
	await expect(applyButtons(page)).toHaveCount(3);

	await page.setViewportSize({ width: 390, height: 844 });
	await openChat(page);
	await expect(applyButtons(page)).toHaveCount(3);
	await screenshot(page, JOURNEY, 'prompt-actions-390');

	await applyButtons(page).nth(0).click();
	await applyButtons(page).nth(1).click();
	await applyButtons(page).nth(2).click();
	await page.getByRole('button', { name: 'Close chat', exact: true }).click();
	await page.getByRole('button', { name: 'Prompt', exact: true }).click();
	await expect(page.locator('[data-prompt-picker]')).toHaveAttribute('aria-label', 'Prompt 1 of 2');
	await screenshot(page, JOURNEY, 'prompt-actions-applied-390');
});
