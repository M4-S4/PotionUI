import { test, expect, type Locator, type Page } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot } from './helpers';
import { startFakeLLM, seedFakeLlmConfig, type FakeLLMServer } from './fake-llm';

const JOURNEY = 'chat-tool-approval';
const BACKEND = process.env.E2E_BACKEND_URL || 'http://127.0.0.1:8055';

let fake: FakeLLMServer;

test.beforeEach(async () => {
	fake = await startFakeLLM();
});

test.afterEach(async () => {
	await fake.close();
});

async function openChatPanel(page: Page) {
	await page.goto('/models');
	const fab = page.getByRole('button', { name: 'AI Chat' });
	await expect(fab).toBeVisible({ timeout: 15000 });
	await fab.click();
	const composer = page.locator('.composer:has(button[aria-label="Send message"])');
	await expect(composer).toBeVisible({ timeout: 15000 });
	return composer;
}

async function sendMessage(page: Page, composer: Locator, text: string) {
	const chipInput = composer.locator('[role="textbox"][aria-placeholder]');
	await expect(chipInput).toHaveAttribute('contenteditable', 'true');
	await chipInput.click();
	await page.keyboard.type(text);
	await composer.locator('button[aria-label="Send message"]').click();
}

async function seedChat(page: Page) {
	await loginAsOwner(page);
	const token = await ownerToken(page);
	await seedFakeLlmConfig(page.request, BACKEND, token, fake.url);
	return token;
}

async function seedMemoryNote(page: Page, token: string, key: string) {
	const response = await page.request.post(`${BACKEND}/api/chat/memory`, {
		headers: { Authorization: `Bearer ${token}` },
		data: { key, content: 'prefers painterly fantasy scenes over photorealism', scope: 'global' }
	});
	expect(response.ok(), `memory note create -> ${response.status()}`).toBeTruthy();
	const body = await response.json();
	expect(body.success, JSON.stringify(body)).toBeTruthy();
	return body.data.id as string;
}

async function memoryNoteIds(page: Page, token: string) {
	const response = await page.request.get(`${BACKEND}/api/chat/memory`, {
		headers: { Authorization: `Bearer ${token}` }
	});
	expect(response.ok()).toBeTruthy();
	const body = await response.json();
	return ((body.data?.notes ?? []) as Array<{ id: string }>).map((n) => n.id);
}

function proposeDelete(noteId: string) {
	fake.enqueue({ kind: 'tool_call', name: 'delete_memory', arguments: { note_id: noteId } });
}

function toolResultsSentToModel(): string {
	return JSON.stringify(fake.requests.filter((r) => r.servedAs === 'queued-turn').map((r) => r.messages));
}

test('approving an inline card applies it and the reply streams below the card', async ({ page }) => {
	test.setTimeout(120000);
	const token = await seedChat(page);
	const noteId = await seedMemoryNote(page, token, 'e2e-approve-style');
	const composer = await openChatPanel(page);

	proposeDelete(noteId);
	await sendMessage(page, composer, 'Forget my style preference');

	const card = page.locator('[data-testid="action-card"]').last();
	await expect(card).toHaveAttribute('data-state', 'pending', { timeout: 30000 });
	await expect(card).toContainText('Needs approval');
	await expect(card).toContainText('Delete memory note');
	await expect(card).toContainText('e2e-approve-style');
	await expect(page.locator('.question-dock')).toHaveCount(0);

	const chipInput = composer.locator('[role="textbox"][aria-placeholder]');
	await expect(chipInput).toHaveAttribute('contenteditable', 'true');
	await expect(chipInput).toHaveAttribute('aria-placeholder', 'Reply, or tell it what to change instead…');

	await screenshot(page, JOURNEY, 'card-pending');

	fake.enqueue({ kind: 'text', text: 'Deleted the e2e-approve-style note from memory.' });
	await card.locator('.action-btn.is-approve').click();

	await expect(card).toHaveAttribute('data-state', 'applied', { timeout: 30000 });
	await expect(card).toContainText('Applied');
	await expect(page.getByText('Deleted the e2e-approve-style note from memory.')).toBeVisible({ timeout: 30000 });
	await expect(page.getByText('(fake-llm: no scripted turn left)')).toHaveCount(0);
	await screenshot(page, JOURNEY, 'card-applied');

	expect(await memoryNoteIds(page, token), 'note is gone after approve').not.toContain(noteId);
});

test('rejecting with a reason sends the reason back and nothing is deleted', async ({ page }) => {
	test.setTimeout(120000);
	const token = await seedChat(page);
	const noteId = await seedMemoryNote(page, token, 'e2e-reject-style');
	const composer = await openChatPanel(page);

	proposeDelete(noteId);
	await sendMessage(page, composer, 'Forget my style preference');

	const card = page.locator('[data-testid="action-card"]').last();
	await expect(card).toHaveAttribute('data-state', 'pending', { timeout: 30000 });

	fake.enqueue({ kind: 'text', text: 'Understood - I kept the e2e-reject-style note.' });
	await card.getByRole('button', { name: 'Reject' }).first().click();
	const reason = card.getByRole('textbox', { name: 'Reason (optional)' });
	await expect(reason).toBeFocused();
	await reason.fill('Keep it until the series is finished');
	await screenshot(page, JOURNEY, 'card-reject-reason');
	await reason.press('Enter');

	await expect(card).toHaveAttribute('data-state', 'rejected', { timeout: 30000 });
	await expect(card).toContainText('"Keep it until the series is finished"');
	await expect(page.getByText('I kept the e2e-reject-style note.')).toBeVisible({ timeout: 30000 });
	await screenshot(page, JOURNEY, 'card-rejected');

	expect(toolResultsSentToModel()).toContain('Keep it until the series is finished');
	expect(await memoryNoteIds(page, token), 'note still exists after reject').toContain(noteId);
});

test('typing a message while a card is pending rejects it with that message as the reason', async ({ page }) => {
	test.setTimeout(120000);
	const token = await seedChat(page);
	const noteId = await seedMemoryNote(page, token, 'e2e-reply-style');
	const composer = await openChatPanel(page);

	proposeDelete(noteId);
	await sendMessage(page, composer, 'Forget my style preference');

	const card = page.locator('[data-testid="action-card"]').last();
	await expect(card).toHaveAttribute('data-state', 'pending', { timeout: 30000 });

	fake.enqueue({ kind: 'text', text: 'Keeping the note. Here are three painterly references instead.' });
	await sendMessage(page, composer, 'Actually keep it, just suggest three references');

	await expect(card).toHaveAttribute('data-state', 'rejected', { timeout: 30000 });
	await expect(card).toContainText('"Actually keep it, just suggest three references"');
	await expect(page.getByText('Here are three painterly references instead.')).toBeVisible({ timeout: 30000 });
	await screenshot(page, JOURNEY, 'card-rejected-by-reply');

	expect(await memoryNoteIds(page, token), 'note still exists after replying instead').toContain(noteId);
	console.log(`[${JOURNEY}] fake-llm requests=${fake.requests.length}, unconsumed turns=${fake.pending()}`);
});
