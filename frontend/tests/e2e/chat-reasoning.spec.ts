import { test, expect, type Page } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot } from './helpers';
import { startFakeLLM, seedFakeLlmConfig, type FakeLLMServer } from './fake-llm';

const JOURNEY = 'chat-reasoning';
const BACKEND = process.env.E2E_BACKEND_URL || 'http://127.0.0.1:8055';
const REASONING = 'Weighing a softer palette against the harsher one before answering.';
const SUMMARY = 'Compared the softer palette with the harsher one and picked the softer.';

const VIEWPORTS = [
	{ label: '1440', width: 1440, height: 900 },
	{ label: '390', width: 390, height: 844 }
] as const;

const THEMES = ['dark', 'light'] as const;

type Kind = 'raw' | 'summary' | 'omitted';

interface Block {
	kind: Kind;
	text: string;
	duration_ms: number;
}

interface Notice {
	kind: string;
	message: string;
}

interface Reply {
	answer: string;
	reasoning?: Block[];
	notices?: Notice[];
}

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

function eventsFor(reply: Reply, index: number): Array<{ type: string; data: unknown }> {
	const now = new Date().toISOString();
	const blocks = reply.reasoning ?? [];
	const notices = reply.notices ?? [];
	const events: Array<{ type: string; data: unknown }> = [
		{ type: 'message_created', data: { user_message_id: `mock-user-${index}` } },
		{
			type: 'status',
			data: {
				step: 'thinking',
				state: 'started',
				...(blocks.length ? { detail: { thinking_visible: true } } : {})
			}
		}
	];
	for (const notice of notices.filter((entry) => entry.kind === 'retrying')) {
		events.push({ type: 'notice', data: { ...notice, detail: { retry_after: 4 } } });
	}
	blocks.forEach((block, position) => {
		if (block.kind !== 'omitted') {
			events.push({ type: 'reasoning', data: { block: position, kind: block.kind, text: block.text } });
		}
		events.push({ type: 'reasoning_end', data: { block: position, duration_ms: block.duration_ms } });
	});
	events.push({ type: 'token', data: { content: reply.answer } });
	events.push({
		type: 'step',
		data: { seq: 1, finish: 'stop', usage: { input_tokens: 5210, output_tokens: 312, reasoning_tokens: 180 } }
	});
	events.push({
		type: 'done',
		data: {
			user_message: { id: `mock-user-${index}`, created_at: now },
			assistant_message: {
				id: `mock-assistant-${index}`,
				role: 'assistant',
				content: reply.answer,
				created_at: now,
				tool_executions: [],
				...(blocks.length
					? {
							reasoning: blocks.map((block) => ({
								kind: block.kind,
								text: block.kind === 'omitted' ? '' : block.text,
								redacted: false,
								duration_ms: block.duration_ms
							}))
						}
					: {}),
				usage: {
					input_tokens: 5210,
					output_tokens: 312,
					reasoning_tokens: 180,
					cached_input_tokens: 4800,
					cache_write_tokens: 0,
					cost: 0.0031
				},
				notices: notices.map((notice, position) => ({ ...notice, step_seq: position + 1 })),
				metadata: {
					behavior_trace: {
						steps: [
							{ step: 'thinking', duration_ms: 5 },
							{ step: 'answering', duration_ms: 5 }
						],
						token_counts: { prompt: null, completion: null },
						...(blocks.length ? { thinking_visible: true } : {})
					}
				}
			}
		}
	});
	return events;
}

async function mockReplies(page: Page, replies: Reply[]) {
	let served = 0;
	await page.route('**/api/chat/sessions/*/messages/stream', async (route) => {
		if (route.request().method() !== 'POST') return route.fallback();
		const index = Math.min(served, replies.length - 1);
		served += 1;
		await route.fulfill({
			status: 200,
			contentType: 'text/event-stream',
			headers: { 'access-control-allow-origin': '*', 'access-control-allow-headers': '*' },
			body: sse(eventsFor(replies[index], index))
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

async function expectNoHorizontalScroll(page: Page) {
	const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
	expect(overflow, 'page has no horizontal scroll').toBeLessThanOrEqual(0);
}

for (const viewport of VIEWPORTS) {
	test.describe(`at ${viewport.label}px`, () => {
		test.use({ viewport: { width: viewport.width, height: viewport.height } });

		test('raw reasoning folds to a timed header above the answer and reopens on click', async ({ page }) => {
			test.setTimeout(120000);
			await openChat(page);
			await mockReplies(page, [
				{ answer: 'Go with the softer palette.', reasoning: [{ kind: 'raw', text: REASONING, duration_ms: 6000 }] }
			]);
			await ask(page, 'Which palette should I use?');

			const block = page.locator('[data-testid="reasoning-block"]').last();
			await expect(block).toBeVisible({ timeout: 30000 });
			const toggle = block.locator('[data-testid="reasoning-toggle"]');
			await expect(toggle).toContainText('Thought for 6s');
			await expect(toggle).toHaveAttribute('aria-expanded', 'false');
			await expect(block.locator('[data-testid="reasoning-body"]')).toHaveCount(0);

			const answer = page.locator('.assistant-copy', { hasText: 'Go with the softer palette.' }).last();
			const blockBox = await block.boundingBox();
			const answerBox = await answer.boundingBox();
			expect(blockBox!.y + blockBox!.height, 'the block sits above the answer').toBeLessThanOrEqual(answerBox!.y + 1);
			await expectNoHorizontalScroll(page);

			await toggle.click();
			await expect(toggle).toHaveAttribute('aria-expanded', 'true');
			await expect(block.locator('[data-testid="reasoning-body"]')).toContainText(REASONING);
			await screenshot(page, JOURNEY, `raw-reopened-${viewport.label}`);

			await toggle.click();
			await expect(block.locator('[data-testid="reasoning-body"]')).toHaveCount(0);
		});

		test('a provider summary is worded as reasoning and labelled as a summary once opened', async ({ page }) => {
			test.setTimeout(120000);
			await openChat(page);
			await mockReplies(page, [
				{ answer: 'Go with the softer palette.', reasoning: [{ kind: 'summary', text: SUMMARY, duration_ms: 1840 }] }
			]);
			await ask(page, 'Which palette should I use?');

			const block = page.locator('[data-testid="reasoning-block"]').last();
			const toggle = block.locator('[data-testid="reasoning-toggle"]');
			await expect(toggle).toContainText('Reasoned for 2s', { timeout: 30000 });
			await expect(toggle).toHaveAttribute('aria-expanded', 'false');

			await toggle.click();
			const body = block.locator('[data-testid="reasoning-body"]');
			await expect(body).toContainText(SUMMARY);
			await expect(body).toContainText('Summary');
			await expectNoHorizontalScroll(page);
			await screenshot(page, JOURNEY, `summary-opened-${viewport.label}`);
		});

		test('hidden reasoning shows a quiet marker with its duration and nothing to open', async ({ page }) => {
			test.setTimeout(120000);
			await openChat(page);
			await mockReplies(page, [
				{ answer: 'Go with the softer palette.', reasoning: [{ kind: 'omitted', text: '', duration_ms: 4200 }] }
			]);
			await ask(page, 'Which palette should I use?');

			const marker = page.locator('[data-testid="reasoning-block"][data-kind="omitted"]').last();
			await expect(marker).toContainText('Reasoned for 4s', { timeout: 30000 });
			await expect(marker.locator('button')).toHaveCount(0);
			await expect(marker.locator('[data-testid="reasoning-body"]')).toHaveCount(0);
			await expectNoHorizontalScroll(page);
			await screenshot(page, JOURNEY, `omitted-${viewport.label}`);
		});

		test('notices from the stream and from the stored message appear above the reply', async ({ page }) => {
			test.setTimeout(120000);
			await openChat(page);
			await mockReplies(page, [
				{
					answer: 'Go with the softer palette.',
					notices: [
						{ kind: 'retrying', message: 'Rate limited; retrying in 4 s.' },
						{ kind: 'reasoning_dropped', message: 'Reasoning made with a different API key was left out.' },
						{ kind: 'structured_repaired', message: 'The structured reply was repaired.' },
						{ kind: 'brand_new_kind', message: 'A kind this build has never seen.' }
					]
				}
			]);
			await ask(page, 'Which palette should I use?');

			const notices = page.locator('[data-testid="chat-notice"]');
			await expect(notices.filter({ hasText: 'Reasoning made with a different API key was left out.' })).toBeVisible({
				timeout: 30000
			});
			await expect(notices.filter({ hasText: 'Rate limited; retrying in 4 s.' })).toBeVisible();
			await expect(notices.filter({ hasText: 'The structured reply was repaired.' })).toBeVisible();
			await expect(notices.filter({ hasText: 'A kind this build has never seen.' })).toBeVisible();

			const answer = page.locator('.assistant-copy', { hasText: 'Go with the softer palette.' }).last();
			const firstBox = await notices.first().boundingBox();
			const answerBox = await answer.boundingBox();
			expect(firstBox!.y + firstBox!.height, 'notices sit above the answer').toBeLessThanOrEqual(answerBox!.y + 1);
			await expectNoHorizontalScroll(page);
			await screenshot(page, JOURNEY, `notices-${viewport.label}`);
		});

		test('the trace names the wait differently while reasoning is shown', async ({ page }) => {
			test.setTimeout(120000);
			await openChat(page);
			await mockReplies(page, [
				{ answer: 'Go with the softer palette.', reasoning: [{ kind: 'raw', text: REASONING, duration_ms: 6000 }] }
			]);
			await ask(page, 'Which palette should I use?');

			const trace = page.locator('.behavior-trace-toggle').last();
			await expect(trace).toContainText('Waiting for the model…', { timeout: 30000 });
		});

		test('a reply without reasoning shows no block and keeps the old wording', async ({ page }) => {
			test.setTimeout(120000);
			await openChat(page);
			await mockReplies(page, [{ answer: 'Go with the softer palette.' }]);
			await ask(page, 'Which palette should I use?');

			const trace = page.locator('.behavior-trace-toggle').last();
			await expect(trace).toContainText('Thinking…', { timeout: 30000 });
			await expect(page.locator('[data-testid="reasoning-block"]')).toHaveCount(0);
			await expect(page.locator('[data-testid="chat-notices"]')).toHaveCount(0);
		});

		for (const theme of THEMES) {
			test(`all three kinds and a notice read cleanly in ${theme}`, async ({ page }) => {
				test.setTimeout(180000);
				await openChat(page);
				await page.evaluate((value) => document.documentElement.setAttribute('data-theme', value), theme);
				await mockReplies(page, [
					{ answer: 'Raw first.', reasoning: [{ kind: 'raw', text: REASONING, duration_ms: 6000 }] },
					{ answer: 'Summary next.', reasoning: [{ kind: 'summary', text: SUMMARY, duration_ms: 1840 }] },
					{
						answer: 'Hidden last.',
						reasoning: [{ kind: 'omitted', text: '', duration_ms: 4200 }],
						notices: [{ kind: 'reasoning_dropped', message: 'Reasoning made with a different API key was left out.' }]
					}
				]);

				await ask(page, 'First question');
				await expect(page.locator('.assistant-copy', { hasText: 'Raw first.' })).toBeVisible({ timeout: 30000 });
				await ask(page, 'Second question');
				await expect(page.locator('.assistant-copy', { hasText: 'Summary next.' })).toBeVisible({ timeout: 30000 });
				await ask(page, 'Third question');
				await expect(page.locator('.assistant-copy', { hasText: 'Hidden last.' })).toBeVisible({ timeout: 30000 });

				await page.locator('[data-testid="reasoning-toggle"]').first().click();
				await page.locator('[data-testid="reasoning-toggle"]').nth(1).click();
				await expect(page.locator('[data-testid="reasoning-body"]')).toHaveCount(2);
				await expectNoHorizontalScroll(page);
				await screenshot(page, JOURNEY, `kinds-${theme}-${viewport.label}`);
			});
		}
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
