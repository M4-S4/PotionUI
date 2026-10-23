import { test, expect, type Page } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot } from './helpers';

const JOURNEY = 'admin-llm-context-window';

const VIEWPORTS = [
	{ name: '1440', width: 1440, height: 900 },
	{ name: '390', width: 390, height: 844 }
];

const BASE = { supported: true, detected: null, detected_kind: null, effective: null, discovered_at: null, error: null };

const STATES = [
	{
		label: 'runtime',
		data: { ...BASE, detected: 32768, detected_kind: 'runtime', effective: 32768 },
		field: '',
		expected: 'Using 32,768 tokens, detected from the server'
	},
	{
		label: 'trained-capped',
		data: { ...BASE, detected: 262144, detected_kind: 'trained', effective: 32768 },
		field: '',
		expected: 'The model supports 262,144; raise the window if your GPU has room'
	},
	{
		label: 'field-set',
		data: { ...BASE, detected: 131072, detected_kind: 'runtime', effective: 4096 },
		field: '4096',
		expected: 'Detected 131,072 tokens'
	},
	{
		label: 'unsupported',
		data: { ...BASE, supported: false },
		field: '',
		expected: "This provider can't report its window — set it here"
	},
	{
		label: 'error',
		data: { ...BASE, error: 'Could not reach the server.' },
		field: '',
		expected: 'Could not reach the server. Using 8,192 tokens until a window is set.'
	}
];

async function mockDiscover(page: Page, data: Record<string, unknown>, calls: { count: number }) {
	await page.route('**/api/llm/context-window/discover', async (route) => {
		calls.count += 1;
		await route.fulfill({ json: { success: true, data } });
	});
}

for (const state of STATES) {
	test(`LLM config context window: ${state.label}`, async ({ page }) => {
		await loginAsOwner(page);
		const token = await ownerToken(page);
		const name = `e2e-ctx-${state.label}-${Date.now()}`;
		const create = await page.request.post('/api/llm/configurations', {
			headers: { Authorization: `Bearer ${token}` },
			data: {
				name,
				type: 'ollama',
				enabled: true,
				base_url: 'http://127.0.0.1:1',
				model: 'llama3',
				system_message: 'You are a helpful assistant.',
				provider_options: state.field ? { num_ctx: Number(state.field) } : {}
			}
		});
		expect(create.ok(), `config create -> ${create.status()}`).toBeTruthy();

		const calls = { count: 0 };
		await mockDiscover(page, state.data, calls);

		await page.goto('/admin?tab=llm');
		await page.getByRole('row').filter({ hasText: name }).click({ timeout: 15000 });

		const field = page.locator('input[id$="-context-window"]');
		await expect(field).toBeVisible();
		await expect(field).toHaveValue(state.field);
		await expect(page.getByTestId('context-window-readout')).toContainText(state.expected);
		expect(calls.count).toBe(1);
		await expect(page.getByText('Context Size')).toHaveCount(0);
		await expect(page.getByText('shared with image generation')).toBeVisible();

		for (const vp of VIEWPORTS) {
			await page.setViewportSize({ width: vp.width, height: vp.height });
			await field.scrollIntoViewIfNeeded();
			await screenshot(page, JOURNEY, `${state.label}-${vp.name}`);
		}

		await page.getByRole('button', { name: 'Detect' }).click();
		await expect.poll(() => calls.count).toBe(2);
	});
}
