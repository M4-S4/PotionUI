import { test, expect, type Page } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot } from './helpers';
import { installAndSelectImagePreset } from './presetPreamble';
import { startFakeLLM, seedFakeLlmConfig, type FakeLLMServer } from './fake-llm';

const JOURNEY = 'chat-segment-add-delete';
const BEAT = 450;
const BACKEND = process.env.E2E_BACKEND_URL || 'http://127.0.0.1:8055';

const FIRST = 'portrait of a lighthouse keeper';
const SECOND = 'weathered hands on a brass lantern';
const THIRD = 'heavy fog, harsh shadows';
const INSERTED = 'soft rim light';
const KNOWN = [FIRST, SECOND, THIRD, INSERTED];

let fake: FakeLLMServer;

test.beforeEach(async () => {
	fake = await startFakeLLM();
});

test.afterEach(async () => {
	await fake.close();
});

async function promptOrder(page: Page): Promise<string[]> {
	const texts = await page.locator('.inline-chip-editor[role="textbox"]').allInnerTexts();
	return texts
		.map((text) => KNOWN.find((known) => text.includes(known)))
		.filter((known): known is string => !!known);
}

async function send(page: Page, text: string) {
	const composer = page.locator('.composer:has(button[aria-label="Send message"])');
	const chipInput = composer.locator('[role="textbox"][aria-placeholder]');
	await expect(chipInput).toHaveAttribute('contenteditable', 'true');
	await chipInput.click();
	await page.keyboard.type(text);
	await composer.locator('button[aria-label="Send message"]').click();
}

function lastReplyButtons(page: Page) {
	return page.locator('.assistant-message').last().locator('[data-testid="segment-action-apply"]');
}

test.use({ viewport: { width: 1440, height: 900 } });

test('assistant adds a segment after #1 and deletes another, and the delete can be undone', async ({ page }) => {
	test.setTimeout(180000);
	await loginAsOwner(page);
	const token = await ownerToken(page);
	await seedFakeLlmConfig(page.request, BACKEND, token, fake.url);

	const preset = await installAndSelectImagePreset(page);
	if (!preset) {
		test.skip(true, 'No native image preset available on this throwaway instance to render the prompt editor with.');
		return;
	}
	await expect(page.locator('.inline-chip-editor[role="textbox"]').first()).toBeVisible({ timeout: 20000 });

	await page.getByRole('button', { name: 'AI Chat' }).click();
	await expect(page.locator('.composer:has(button[aria-label="Send message"])')).toBeVisible({ timeout: 15000 });

	fake.enqueue({
		kind: 'text',
		text:
			'Three segments to start from:\n' +
			`<tool_action type="add_segment">${FIRST}</tool_action>\n` +
			`<tool_action type="add_segment">${SECOND}</tool_action>\n` +
			`<tool_action type="add_segment">${THIRD}</tool_action>`
	});
	await send(page, 'Build me a starting prompt');

	await expect(page.getByText('Add Segment').first()).toBeVisible({ timeout: 30000 });
	await expect(lastReplyButtons(page)).toHaveCount(3);
	for (let i = 0; i < 3; i++) {
		await lastReplyButtons(page).nth(i).click();
	}
	await expect(lastReplyButtons(page).nth(2)).toHaveText(/Added/);
	await expect(lastReplyButtons(page).nth(2)).toBeDisabled();
	await expect.poll(() => promptOrder(page)).toEqual([FIRST, SECOND, THIRD]);
	await page.waitForTimeout(BEAT);
	await screenshot(page, JOURNEY, 'three-segments-added');

	fake.enqueue({
		kind: 'text',
		text:
			'Drop the harsh shadows and add a rim light after the first segment:\n' +
			'<tool_action type="delete_segment" segment_index="2"></tool_action>\n' +
			`<tool_action type="add_segment" segment_index="0">${INSERTED}</tool_action>`
	});
	await send(page, 'Soften the lighting');

	const reply = page.locator('.assistant-message').last();
	await expect(reply.getByText('Delete Segment #3')).toBeVisible({ timeout: 30000 });
	await expect(reply.getByText('Add Segment')).toBeVisible();
	await expect(reply.locator('[data-testid="segment-removed-text"]')).toContainText(THIRD);
	await page.waitForTimeout(BEAT);
	await screenshot(page, JOURNEY, 'delete-and-add-proposed');

	await lastReplyButtons(page).nth(0).click();
	await expect(lastReplyButtons(page).nth(0)).toHaveText(/Deleted/);
	await expect.poll(() => promptOrder(page)).toEqual([FIRST, SECOND]);
	await expect(reply.locator('[data-testid="segment-action-undo"]')).toBeVisible();
	await expect(reply.locator('[data-testid="segment-removed-text"]')).toContainText(THIRD);

	await lastReplyButtons(page).nth(1).click();
	await expect(lastReplyButtons(page).nth(1)).toHaveText(/Added/);
	await expect.poll(() => promptOrder(page)).toEqual([FIRST, INSERTED, SECOND]);
	await page.waitForTimeout(BEAT);
	await screenshot(page, JOURNEY, 'delete-and-add-applied-1440');

	await reply.locator('[data-testid="segment-action-undo"]').click();
	await expect.poll(() => promptOrder(page)).toEqual([FIRST, INSERTED, SECOND, THIRD]);
	await expect(reply.locator('[data-testid="segment-action-undo"]')).toHaveCount(0);
	await expect(lastReplyButtons(page).nth(0)).toHaveText('Delete');
	await page.waitForTimeout(BEAT);
	await screenshot(page, JOURNEY, 'delete-undone');

	await page.setViewportSize({ width: 390, height: 844 });
	await page.waitForTimeout(BEAT);
	await screenshot(page, JOURNEY, 'delete-undone-390');

	console.log(
		`[${JOURNEY}] preset ${preset.id}: three adds appended, delete+add applied by index, delete undone; ` +
			`fake-llm requests=${fake.requests.length}, unconsumed turns=${fake.pending()}`
	);
});
