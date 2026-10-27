import { test, expect, type Page, type Route } from '@playwright/test';
import { loginAsOwner, screenshot, fulfillJson } from './helpers';

const JOURNEY = 'admin-llm-profile';

const VIEWPORTS = [
	{ label: '1440', width: 1440, height: 900 },
	{ label: '390', width: 390, height: 844 }
] as const;

const THEMES = ['dark', 'light'] as const;

const CONFIG_NAME = 'Claude Sonnet (mock)';

const CONFIG = {
	id: 'c1',
	name: CONFIG_NAME,
	type: 'anthropic',
	enabled: true,
	base_url: 'https://api.anthropic.com',
	api_key_set: true,
	model: 'claude-sonnet-5-5',
	system_message: 'You are a helpful assistant.',
	temperature: 0.7,
	max_tokens: 1000,
	timeout: 30,
	supports_vision: true,
	memory_reflection: true,
	provider_options: {}
};

const TYPES = [
	{ type: 'anthropic', label: 'Anthropic', adapter: 'anthropic-messages', default_base_url: 'https://api.anthropic.com', supports_model_listing: true, source: 'core' },
	{ type: 'openai', label: 'OpenAI', adapter: 'openai-responses', default_base_url: 'https://api.openai.com/v1', supports_model_listing: true, source: 'core' },
	{ type: 'openai-compatible', label: 'OpenAI-compatible server', adapter: 'openai-chat', default_base_url: null, supports_model_listing: true, source: 'core' },
	{ type: 'gemini', label: 'Gemini', adapter: 'gemini-interactions', default_base_url: null, supports_model_listing: true, source: 'core' },
	{ type: 'openrouter', label: 'OpenRouter', adapter: 'openrouter', default_base_url: 'https://openrouter.ai/api/v1', supports_model_listing: true, source: 'core' },
	{ type: 'ollama', label: 'Ollama', adapter: 'ollama-chat', default_base_url: 'http://localhost:11434', supports_model_listing: true, source: 'core' },
	{ type: 'native', label: 'Native', adapter: 'native', default_base_url: null, supports_model_listing: false, source: 'core' },
	{ type: 'deepseek', label: 'DeepSeek', adapter: 'openai-chat', default_base_url: 'https://api.deepseek.com', supports_model_listing: true, source: 'deepseek-provider' }
];

const SCHEMA = {
	fields: [
		{ name: 'reasoning_kind', kind: 'enum', options: ['none', 'toggle', 'effort', 'budget', 'always_on'], label: 'Reasoning control', group: 'reasoning' },
		{ name: 'replay', kind: 'enum', options: ['send_all', 'drop_earlier', 'preserve', 'never'], label: 'Reasoning replay', group: 'reasoning' },
		{ name: 'interleaved_thinking', kind: 'bool', label: 'Interleaved thinking', group: 'reasoning' },
		{ name: 'tool_choice', kind: 'set', options: ['auto', 'none', 'required', 'named'], label: 'Tool choice modes', group: 'tools' },
		{ name: 'parallel_tools', kind: 'bool', label: 'Parallel tool calls', group: 'tools' },
		{ name: 'vision', kind: 'bool', label: 'Vision input', group: 'input' },
		{ name: 'context_window', kind: 'int', min: 1024, label: 'Context window', group: 'limits' },
		{ name: 'max_output', kind: 'int', min: 1, label: 'Max output tokens', group: 'limits' },
		{ name: 'reasoning_field', kind: 'string', label: 'Reasoning field name', group: 'wire' }
	]
};

function field(value: unknown, source: 'seed' | 'discovered' | 'override', seed: unknown, discovered: unknown, override: unknown) {
	return { value, source, seed, discovered, override };
}

type MockField = ReturnType<typeof field>;

interface MockProfile {
	fields: Record<string, MockField>;
	[key: string]: unknown;
}

function profile(overrides: Record<string, MockField> = {}, extra: Record<string, unknown> = {}): MockProfile {
	return {
		config_id: 'c1',
		adapter: 'anthropic-messages',
		model: 'claude-sonnet-5-5',
		discovered_at: new Date(Date.now() - 3 * 3600 * 1000).toISOString(),
		discovery_error: null,
		fields: {
			reasoning_kind: field('budget', 'discovered', 'effort', 'budget', null),
			replay: field('preserve', 'override', 'send_all', null, 'preserve'),
			interleaved_thinking: field(true, 'seed', true, null, null),
			tool_choice: field(['auto', 'none', 'required'], 'seed', ['auto', 'none', 'required'], null, null),
			parallel_tools: field(true, 'seed', true, null, null),
			vision: field(true, 'discovered', false, true, null),
			context_window: field(200000, 'discovered', 128000, 200000, null),
			max_output: field(64000, 'discovered', 32000, 64000, null),
			reasoning_field: field('thinking', 'seed', 'thinking', null, null),
			...overrides
		},
		verified: { suite: 'chat_eval:v3', passed_at: '2026-10-27', runtime: 'anthropic api 2026-10', gpu_tier: 'hosted' },
		seed_rows: [
			{ id: 'base', root: 'core', file: 'profile_base.yml' },
			{ id: 'anthropic-messages', root: 'marketplace', file: 'anthropic.yml' },
			{ id: 'mine', root: 'local', file: 'mine.yml' }
		],
		...extra
	};
}

interface Calls {
	put: Array<Record<string, unknown>>;
	refresh: number;
}

async function mockLlmApi(page: Page, state: { profile: MockProfile }): Promise<Calls> {
	const calls: Calls = { put: [], refresh: 0 };
	await page.route('**/api/llm/**', async (route: Route) => {
		const request = route.request();
		const path = new URL(request.url()).pathname;
		const method = request.method();
		if (method === 'GET' && path === '/api/llm/configurations') {
			return fulfillJson(route, { success: true, data: { configurations: [CONFIG] } });
		}
		if (method === 'GET' && path === '/api/llm/types') {
			return fulfillJson(route, { success: true, data: { types: TYPES } });
		}
		if (method === 'GET' && path === '/api/llm/assignment-summary') {
			return fulfillJson(route, { success: true, data: {} });
		}
		if (method === 'GET' && path === '/api/llm/profile/schema') {
			return fulfillJson(route, { success: true, data: SCHEMA });
		}
		if (method === 'GET' && path === '/api/llm/configurations/c1/profile') {
			return fulfillJson(route, { success: true, data: state.profile });
		}
		if (method === 'PUT' && path === '/api/llm/configurations/c1/profile') {
			const overrides = (request.postDataJSON() as { overrides: Record<string, unknown> }).overrides;
			calls.put.push(overrides);
			const fields: Record<string, MockField> = { ...state.profile.fields };
			for (const [name, value] of Object.entries(overrides)) {
				const current = fields[name];
				fields[name] =
					value === null
						? field(current.discovered ?? current.seed, current.discovered != null ? 'discovered' : 'seed', current.seed, current.discovered, null)
						: field(value, 'override', current.seed, current.discovered, value);
			}
			state.profile = { ...state.profile, fields };
			return fulfillJson(route, { success: true, data: state.profile });
		}
		if (method === 'POST' && path === '/api/llm/configurations/c1/profile/refresh') {
			calls.refresh += 1;
			state.profile = { ...state.profile, discovered_at: new Date().toISOString() };
			return fulfillJson(route, { success: true, data: state.profile });
		}
		if (method === 'POST' && path === '/api/llm/models/discover') {
			return fulfillJson(route, { success: true, data: { supported: false, models: [], detected: null, error: null } });
		}
		return fulfillJson(route, { success: false, message: `unmocked ${method} ${path}` }, 404);
	});
	await page.route('**/api/chat/pre-actions', (route) => fulfillJson(route, { success: true, data: { actions: [] } }));
	return calls;
}

async function openProfile(page: Page) {
	await page.goto('/admin?tab=llm');
	const row = page.getByRole('row').filter({ hasText: CONFIG_NAME });
	if ((await row.count()) === 0) {
		await page.getByText(CONFIG_NAME).locator('visible=true').first().click({ timeout: 15000 });
	} else {
		await row.click({ timeout: 15000 });
	}
	await page.getByRole('navigation', { name: 'LLM configuration details' }).getByRole('button', { name: 'Profile' }).click();
	await expect(page.locator('[data-testid="profile-row"]').first()).toBeVisible({ timeout: 15000 });
}

const rowFor = (page: Page, name: string) => page.locator(`[data-testid="profile-row"][data-field="${name}"]`);

async function expectNoHorizontalScroll(page: Page) {
	const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
	expect(overflow, 'page has no horizontal scroll').toBeLessThanOrEqual(0);
}

for (const viewport of VIEWPORTS) {
	test.describe(`at ${viewport.label}px`, () => {
		test.use({ viewport: { width: viewport.width, height: viewport.height } });

		test('the type picker lists the types the server offers', async ({ page }) => {
			await loginAsOwner(page);
			await mockLlmApi(page, { profile: profile() });
			await page.goto('/admin?tab=llm');
			const row = page.getByRole('row').filter({ hasText: CONFIG_NAME });
			if ((await row.count()) === 0) {
				await page.getByText(CONFIG_NAME).locator('visible=true').first().click({ timeout: 15000 });
			} else {
				await row.click({ timeout: 15000 });
			}

			const select = page.locator('#edit-config-type');
			await expect(select).toBeVisible({ timeout: 15000 });
			await expect(select.locator('option')).toHaveText(TYPES.map((entry) => entry.label));
			await expect(select).toHaveValue('anthropic');
			await screenshot(page, JOURNEY, `type-picker-${viewport.label}`);
		});

		test('every field shows its value with where it came from, and the verified badge', async ({ page }) => {
			await loginAsOwner(page);
			await mockLlmApi(page, { profile: profile() });
			await openProfile(page);

			await expect(rowFor(page, 'replay')).toHaveAttribute('data-source', 'override');
			await expect(rowFor(page, 'context_window')).toHaveAttribute('data-source', 'discovered');
			await expect(rowFor(page, 'reasoning_field')).toHaveAttribute('data-source', 'seed');
			await expect(rowFor(page, 'context_window').locator('input')).toHaveValue('200000');
			await expect(rowFor(page, 'context_window').locator('[data-testid="profile-layers"]')).toContainText('Seed 128,000');
			await expect(rowFor(page, 'tool_choice').locator('button[data-option="required"]')).toHaveAttribute('aria-pressed', 'true');

			const verified = page.locator('[data-testid="profile-verified"]');
			await expect(verified).toContainText('Verified');
			const seedRows = page.locator('[data-testid="profile-seed-row"]');
			await expect(seedRows).toHaveText(['base · core/profile_base.yml', 'anthropic-messages · marketplace/anthropic.yml', 'mine · local/mine.yml']);
			await expect(page.getByText('chat_eval:v3')).toBeVisible();
			await expect(page.getByText('anthropic api 2026-10')).toBeVisible();
			await expect(page.getByText('hosted', { exact: true })).toBeVisible();
			await expectNoHorizontalScroll(page);
			await screenshot(page, JOURNEY, `profile-${viewport.label}`);
		});

		test('an override is edited by kind, saved as a patch and shown with its new source', async ({ page }) => {
			await loginAsOwner(page);
			const calls = await mockLlmApi(page, { profile: profile() });
			await openProfile(page);

			await rowFor(page, 'replay').locator('select').selectOption('never');
			await rowFor(page, 'tool_choice').locator('button[data-option="named"]').click();
			await rowFor(page, 'context_window').locator('input').fill('262144');
			await expect(page.locator('[data-detail-footer]')).toContainText('3 unsaved changes');
			await screenshot(page, JOURNEY, `profile-edited-${viewport.label}`);

			await page.getByRole('button', { name: 'Save', exact: true }).click();
			await expect(page.locator('[data-detail-footer]')).not.toContainText('unsaved');

			expect(calls.put).toEqual([
				{ replay: 'never', tool_choice: ['auto', 'none', 'required', 'named'], context_window: 262144 }
			]);
			await expect(rowFor(page, 'context_window')).toHaveAttribute('data-source', 'override');
			await expect(rowFor(page, 'replay').locator('select')).toHaveValue('never');
		});

		test('clearing an override sends null and falls back to the layer below', async ({ page }) => {
			await loginAsOwner(page);
			const calls = await mockLlmApi(page, { profile: profile() });
			await openProfile(page);

			await rowFor(page, 'replay').getByRole('button', { name: /Clear override/ }).click();
			await expect(rowFor(page, 'replay')).toHaveAttribute('data-source', 'seed');
			await page.getByRole('button', { name: 'Save', exact: true }).click();
			await expect(page.locator('[data-detail-footer]')).not.toContainText('unsaved');

			expect(calls.put).toEqual([{ replay: null }]);
		});

		test('a value below the schema minimum is explained and cannot be saved', async ({ page }) => {
			await loginAsOwner(page);
			const calls = await mockLlmApi(page, { profile: profile() });
			await openProfile(page);

			await rowFor(page, 'context_window').locator('input').fill('512');

			await expect(rowFor(page, 'context_window').locator('[data-testid="profile-problem"]')).toHaveText('Minimum is 1,024');
			await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled();
			expect(calls.put).toEqual([]);
		});

		test('refresh re-runs discovery without losing an edit', async ({ page }) => {
			await loginAsOwner(page);
			const calls = await mockLlmApi(page, { profile: profile() });
			await openProfile(page);

			await rowFor(page, 'replay').locator('select').selectOption('never');
			await page.getByRole('button', { name: 'Refresh', exact: true }).click();

			await expect.poll(() => calls.refresh).toBe(1);
			await expect(rowFor(page, 'replay').locator('select')).toHaveValue('never');
			await expect(page.locator('[data-detail-footer]')).toContainText('1 unsaved change');
		});

		test('a failed discovery says why and a profile with no verification says so', async ({ page }) => {
			await loginAsOwner(page);
			await mockLlmApi(page, {
				profile: profile({}, { discovery_error: 'GET /v1/models returned 401', discovered_at: null, verified: null })
			});
			await openProfile(page);

			await expect(page.locator('[data-testid="profile-discovery-error"]')).toHaveText('GET /v1/models returned 401');
			await expect(page.locator('[data-testid="profile-unverified"]')).toBeVisible();
			await expectNoHorizontalScroll(page);
			await screenshot(page, JOURNEY, `profile-discovery-failed-${viewport.label}`);
		});

		for (const theme of THEMES) {
			test(`the profile reads cleanly in ${theme}`, async ({ page }) => {
				await loginAsOwner(page);
				await mockLlmApi(page, { profile: profile() });
				await openProfile(page);
				await page.evaluate((value) => document.documentElement.setAttribute('data-theme', value), theme);

				await rowFor(page, 'context_window').locator('input').fill('512');
				await expect(rowFor(page, 'context_window').locator('[data-testid="profile-problem"]')).toBeVisible();
				await expectNoHorizontalScroll(page);
				await screenshot(page, JOURNEY, `profile-${theme}-${viewport.label}`);
			});
		}
	});
}
