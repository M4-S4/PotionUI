import { test, expect, type Page } from '@playwright/test';
import { loginAsOwner, screenshot, fulfillJson } from './helpers';

const JOURNEY = 'admin-chat-trace-usage';

const VIEWPORTS = [
	{ label: '1440', width: 1440, height: 900 },
	{ label: '390', width: 390, height: 844 }
] as const;

const THEMES = ['dark', 'light'] as const;

const NOW = new Date().toISOString();

const SESSION = {
	id: 's1',
	user_id: 'u1',
	mode: 'generation',
	name: 'Palette question',
	status: 'active',
	llm_config_id: 'c1',
	created_at: NOW,
	updated_at: NOW,
	username: 'e2e-owner',
	email: 'owner@example.invalid',
	message_count: 2,
	original_text: null,
	title_generated: true,
	closed_at: null,
	metadata: null,
	messages: [
		{
			id: 'm1',
			session_id: 's1',
			role: 'user',
			content: 'Which palette should I use?',
			parsed_content: null,
			created_at: NOW,
			tokens_used: null,
			prompt_tokens: null,
			completion_tokens: null,
			tool_executions: null,
			metadata: null
		},
		{
			id: 'm2',
			session_id: 's1',
			role: 'assistant',
			content: 'Go with the softer palette.',
			parsed_content: null,
			created_at: NOW,
			tokens_used: 5522,
			prompt_tokens: 5210,
			completion_tokens: 312,
			tool_executions: null,
			metadata: null
		}
	]
};

function trace(id: string, extra: Record<string, unknown>) {
	return {
		id,
		session_id: 's1',
		user_id: 'u1',
		message_id: 'm2',
		purpose: 'chat_tools',
		iteration: 1,
		provider: 'anthropic',
		model: 'claude-sonnet-5-5',
		request_system: 'You are a helpful assistant.',
		request_messages: [{ role: 'user', content: 'Which palette should I use?' }],
		request_params: { max_tokens: 1000 },
		request_tools: ['search_gallery'],
		response_text: 'Go with the softer palette.',
		response_tool_calls: null,
		prompt_tokens: 5210,
		completion_tokens: 312,
		duration_ms: 1840,
		created_at: NOW,
		...extra
	};
}

const TRACES = [
	trace('t1', {
		adapter: 'anthropic-messages',
		usage: {
			input_tokens: 5210,
			output_tokens: 312,
			reasoning_tokens: 180,
			cached_input_tokens: 4800,
			cache_write_tokens: 0,
			cost: 0.0031
		},
		cost: 0.0031,
		request_body: { model: 'claude-sonnet-5-5', max_tokens: 1000, messages: [{ role: 'user', content: 'Which palette should I use?' }] },
		response_raw: 'event: message_start\ndata: {"type":"message_start"}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n'
	}),
	trace('t0', { iteration: 0, purpose: 'title', message_id: null })
];

async function mockSessions(page: Page) {
	await page.route('**/api/chat/admin/sessions**', async (route) => {
		const path = new URL(route.request().url()).pathname;
		if (path === '/api/chat/admin/sessions/s1') {
			return fulfillJson(route, { success: true, data: { session: SESSION, traces: TRACES } });
		}
		if (path === '/api/chat/admin/sessions') {
			return fulfillJson(route, {
				success: true,
				data: { sessions: [SESSION], total: 1, limit: 20, offset: 0, tracing_enabled: true }
			});
		}
		return route.fallback();
	});
}

async function openTrace(page: Page) {
	await page.goto('/admin?tab=llm&view=sessions');
	const row = page.getByRole('row').filter({ hasText: 'Palette question' });
	if ((await row.count()) === 0) {
		await page.getByText('Palette question').locator('visible=true').first().click({ timeout: 15000 });
	} else {
		await row.click({ timeout: 15000 });
	}
	const summary = page.locator('[data-testid="trace-summary"]').first();
	await expect(summary).toBeVisible({ timeout: 15000 });
	return summary;
}

for (const viewport of VIEWPORTS) {
	test.describe(`at ${viewport.label}px`, () => {
		test.use({ viewport: { width: viewport.width, height: viewport.height } });

		test('the call header shows cached tokens, reasoning tokens and cost', async ({ page }) => {
			await loginAsOwner(page);
			await mockSessions(page);
			const summary = await openTrace(page);

			await expect(summary).toContainText('anthropic-messages');
			await expect(summary).toContainText('5,210→312 tok');
			await expect(summary).toContainText('4,800 cached');
			await expect(summary).toContainText('180 reasoning');
			await expect(summary).toContainText('$0.0031');
		});

		test('an opened call lists its usage and the wire request and raw response', async ({ page }) => {
			await loginAsOwner(page);
			await mockSessions(page);
			const summary = await openTrace(page);

			await summary.click();
			const usage = page.locator('[data-testid="trace-usage"]');
			await expect(usage).toBeVisible();
			for (const [label, value] of [
				['Input', '5,210'],
				['Output', '312'],
				['Cached input', '4,800'],
				['Cache write', '0'],
				['Reasoning', '180'],
				['Cost', '$0.0031']
			]) {
				await expect(usage).toContainText(label);
				await expect(usage).toContainText(value);
			}
			await expect(page.getByText('Wire request')).toBeVisible();
			await expect(page.getByText('Raw response')).toBeVisible();
			const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
			expect(overflow, 'page has no horizontal scroll').toBeLessThanOrEqual(0);
			await screenshot(page, JOURNEY, `trace-open-${viewport.label}`);
		});

		test('a call recorded before usage existed keeps its plain token line', async ({ page }) => {
			await loginAsOwner(page);
			await mockSessions(page);
			await openTrace(page);

			const old = page.locator('[data-testid="trace-summary"]').nth(1);
			await expect(old).toContainText('5,210→312 tok');
			await expect(old).not.toContainText('cached');
			await expect(old).not.toContainText('$');
		});

		for (const theme of THEMES) {
			test(`the opened call reads cleanly in ${theme}`, async ({ page }) => {
				await loginAsOwner(page);
				await mockSessions(page);
				const summary = await openTrace(page);
				await page.evaluate((value) => document.documentElement.setAttribute('data-theme', value), theme);

				await summary.click();
				await expect(page.locator('[data-testid="trace-usage"]')).toBeVisible();
				await screenshot(page, JOURNEY, `trace-${theme}-${viewport.label}`);
			});
		}
	});
}
