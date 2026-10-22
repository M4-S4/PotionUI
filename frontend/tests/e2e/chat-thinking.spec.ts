import { test, expect, type Page } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot } from './helpers';
import { startFakeLLM, seedFakeLlmConfig, type FakeLLMServer } from './fake-llm';

const JOURNEY = 'chat-thinking';
const BACKEND = process.env.E2E_BACKEND_URL || 'http://127.0.0.1:8055';
const REASONING = 'Weighing a softer palette against the harsher one before answering.';

const VIEWPORTS = [
	{ label: '1440', width: 1440, height: 900 },
	{ label: '390', width: 390, height: 844 }
] as const;

let fake: FakeLLMServer;

test.beforeEach(async () => {
	fake = await startFakeLLM();
});

test.afterEach(async () => {
	await fake.close();
});

function sse(events: Array<{ type: string; data: unknown }>): string {
	return events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event.data)}\n\n`).join('');
}

async function mockReply(page: Page, options: { withThinking: boolean }) {
	await page.route('**/api/chat/sessions/*/messages/stream', async (route) => {
		if (route.request().method() !== 'POST') return route.fallback();
		const now = new Date().toISOString();
		const events: Array<{ type: string; data: unknown }> = [
			{ type: 'message_created', data: { user_message_id: 'mock-user' } },
			{
				type: 'status',
				data: {
					step: 'thinking',
					state: 'started',
					...(options.withThinking ? { detail: { thinking_visible: true } } : {})
				}
			}
		];
		if (options.withThinking) {
			events.push({ type: 'thinking', data: { content: REASONING, segment: 0 } });
		}
		events.push({ type: 'token', data: { content: 'Go with the softer palette.' } });
		events.push({
			type: 'done',
			data: {
				user_message: { id: 'mock-user', created_at: now },
				assistant_message: {
					id: 'mock-assistant',
					role: 'assistant',
					content: 'Go with the softer palette.',
					created_at: now,
					tool_executions: [],
					metadata: {
						behavior_trace: {
							steps: [
								{ step: 'thinking', duration_ms: 5 },
								{ step: 'answering', duration_ms: 5 }
							],
							token_counts: { prompt: null, completion: null },
							...(options.withThinking ? { thinking_visible: true } : {})
						},
						...(options.withThinking
							? { thinking: { segments: [{ text: REASONING, duration_ms: 6000 }] } }
							: {})
					}
				}
			}
		});
		await route.fulfill({
			status: 200,
			contentType: 'text/event-stream',
			headers: { 'access-control-allow-origin': '*', 'access-control-allow-headers': '*' },
			body: sse(events)
		});
	});
}

async function openChat(page: Page) {
	await loginAsOwner(page);
	const token = await ownerToken(page);
	await seedFakeLlmConfig(page.request, BACKEND, token, fake.url);
	const phone = (page.viewportSize()?.width ?? 1440) < 768;
	if (phone) {
		await page.goto('/generate');
		await page.getByRole('button', { name: 'Open AI chat' }).click();
	} else {
		await page.goto('/models');
		await page.getByRole('button', { name: 'AI Chat' }).click();
	}
	const composer = page.locator('.composer:has(button[aria-label="Send message"])');
	await expect(composer).toBeVisible({ timeout: 15000 });
	return composer;
}

async function ask(page: Page, text: string) {
	const composer = page.locator('.composer:has(button[aria-label="Send message"])');
	const chipInput = composer.locator('[role="textbox"][aria-placeholder]');
	await expect(chipInput).toHaveAttribute('contenteditable', 'true');
	await chipInput.click();
	await page.keyboard.type(text);
	await composer.locator('button[aria-label="Send message"]').click();
}

for (const viewport of VIEWPORTS) {
	test.describe(`at ${viewport.label}px`, () => {
		test.use({ viewport: { width: viewport.width, height: viewport.height } });

		test('a reply with thinking folds to a timed header above the answer and reopens on click', async ({ page }) => {
			test.setTimeout(120000);
			await openChat(page);
			await mockReply(page, { withThinking: true });
			await ask(page, 'Which palette should I use?');

			const block = page.locator('[data-testid="thinking-block"]').last();
			await expect(block).toBeVisible({ timeout: 30000 });
			const toggle = block.locator('[data-testid="thinking-toggle"]');
			await expect(toggle).toContainText('Thought for 6s');
			await expect(toggle).toHaveAttribute('aria-expanded', 'false');
			await expect(block.locator('[data-testid="thinking-body"]')).toHaveCount(0);

			const answer = page.locator('.assistant-copy', { hasText: 'Go with the softer palette.' }).last();
			const blockBox = await block.boundingBox();
			const answerBox = await answer.boundingBox();
			expect(blockBox!.y + blockBox!.height, 'the block sits above the answer').toBeLessThanOrEqual(answerBox!.y + 1);
			const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
			expect(overflow, 'page has no horizontal scroll').toBeLessThanOrEqual(0);

			await toggle.click();
			await expect(toggle).toHaveAttribute('aria-expanded', 'true');
			await expect(block.locator('[data-testid="thinking-body"]')).toContainText(REASONING);
			await screenshot(page, JOURNEY, `reopened-${viewport.label}`);

			await toggle.click();
			await expect(block.locator('[data-testid="thinking-body"]')).toHaveCount(0);
		});

		test('the trace names the wait differently while thinking is shown', async ({ page }) => {
			test.setTimeout(120000);
			await openChat(page);
			await mockReply(page, { withThinking: true });
			await ask(page, 'Which palette should I use?');

			const trace = page.locator('.behavior-trace-toggle').last();
			await expect(trace).toContainText('Waiting for the model…', { timeout: 30000 });
		});

		test('a reply without thinking shows no block and keeps the old wording', async ({ page }) => {
			test.setTimeout(120000);
			await openChat(page);
			await mockReply(page, { withThinking: false });
			await ask(page, 'Which palette should I use?');

			const trace = page.locator('.behavior-trace-toggle').last();
			await expect(trace).toContainText('Thinking…', { timeout: 30000 });
			await expect(page.locator('[data-testid="thinking-block"]')).toHaveCount(0);
		});
	});
}

test('the admin switch saves Show thinking to users on the configuration', async ({ page }) => {
	test.setTimeout(120000);
	await page.setViewportSize({ width: 1440, height: 900 });
	await loginAsOwner(page);
	const token = await ownerToken(page);
	const headers = { Authorization: `Bearer ${token}` };
	const name = `e2e-show-thinking-${Date.now()}`;
	const create = await page.request.post('/api/llm/configurations', {
		headers,
		data: {
			name,
			type: 'ollama',
			enabled: true,
			base_url: 'http://127.0.0.1:1',
			model: 'qwen3',
			system_message: 'You are a helpful assistant.'
		}
	});
	expect(create.ok(), `config create -> ${create.status()}`).toBeTruthy();
	const id = (await create.json()).data?.id as string;

	try {
		await page.goto('/admin?tab=llm');
		await page.getByRole('row').filter({ hasText: name }).click({ timeout: 15000 });

		const toggle = page.locator('input[role="switch"][aria-label="Show thinking to users"]');
		await expect(toggle).not.toBeChecked({ timeout: 15000 });
		await toggle.click();
		await expect(toggle).toBeChecked();

		const saved = page.waitForResponse(
			(response) => response.url().includes(`/api/llm/configurations/${id}`) && response.request().method() === 'PUT'
		);
		await page.getByRole('button', { name: 'Save', exact: true }).click();
		const response = await saved;
		expect(response.request().postDataJSON().show_thinking).toBe(true);

		const stored = await page.request.get(`/api/llm/configurations/${id}`, { headers });
		expect((await stored.json()).data.show_thinking).toBe(true);
		await screenshot(page, JOURNEY, 'admin-switch');
	} finally {
		await page.request.delete(`/api/llm/configurations/${id}`, { headers });
	}
});
