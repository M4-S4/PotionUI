import { test, expect, type Page } from '@playwright/test';
import { fulfillJson, screenshot, SIGNED_OUT } from './helpers';

const JOURNEY = 'onboarding-guide';
const GIB = 1024 ** 3;

const MACHINE = {
	gpu: { name: 'NVIDIA GeForce RTX 4090', vram_bytes: 23.6 * GIB, compute_capability: '8.9', supported: true },
	ram_total_bytes: 64 * GIB,
	models_disk: { path: '/data/potionui/models', free_bytes: 412 * GIB, total_bytes: 900 * GIB }
};

function recipe(overrides: Record<string, unknown>) {
	return {
		id: 'r',
		name: 'R',
		summary: 'summary',
		description: '',
		engine: 'native',
		category: 'image',
		artifact_count: 3,
		total_download_bytes: 10 * GIB,
		presets: [],
		last_completed_at: null,
		blurb: 'blurb',
		min_vram_gb: null,
		recommended_vram_gb: null,
		sources: ['huggingface.co'],
		first_run: true,
		...overrides
	};
}

const RECIPES = [
	recipe({
		id: 'qwen-image-starter',
		name: 'Qwen-Image Starter',
		blurb: 'Strong at readable text inside images. The biggest download here.',
		recommended_vram_gb: 16,
		total_download_bytes: 28 * GIB
	}),
	recipe({
		id: 'sdxl-starter',
		name: 'SDXL Starter',
		blurb: 'The classic all-rounder.',
		min_vram_gb: 8,
		total_download_bytes: 6.62 * GIB
	}),
	recipe({
		id: 'zimage-starter',
		name: 'Z-Image Starter',
		blurb: 'Fast and compact.',
		min_vram_gb: 12,
		recommended_vram_gb: 16,
		total_download_bytes: 19.27 * GIB
	})
];

const NOT_READY = {
	overall: 'not_ready',
	checks: [
		{ area: 'content', status: 'not_ready', code: 'no_presets', message: 'No presets are installed.', action: null }
	]
};

async function noHorizontalOverflow(page: Page) {
	const overflow = await page.evaluate(() => {
		const el = document.scrollingElement!;
		return el.scrollWidth - el.clientWidth;
	});
	expect(overflow, 'page must not scroll horizontally').toBeLessThanOrEqual(0);
}

async function mockGuideApi(page: Page, consentRun?: unknown) {
	await page.addInitScript(() => {
		for (const key of Object.keys(localStorage)) {
			if (key.startsWith('potionui:setup:')) localStorage.removeItem(key);
		}
	});
	await page.route('**/api/readiness*', (route) => fulfillJson(route, NOT_READY));
	await page.route('**/api/setup/machine', (route) => fulfillJson(route, MACHINE));
	await page.route('**/api/setup/recipes', (route) => fulfillJson(route, { recipes: RECIPES }));
	await page.route('**/api/setup/runs/active', (route) => fulfillJson(route, { detail: 'none' }, 404));
	if (consentRun) {
		await page.route('**/api/setup/runs', (route) => {
			if (route.request().method() === 'POST') return fulfillJson(route, consentRun);
			return route.fallback();
		});
		await page.route('**/api/setup/runs/run-1', (route) => fulfillJson(route, consentRun));
	}
}

test.describe('claim explainer', () => {
	test.use({ storageState: SIGNED_OUT });

	for (const viewport of [
		{ name: '1440', width: 1440, height: 900 },
		{ name: '390', width: 390, height: 844 }
	]) {
		test(`welcome then account explainer at ${viewport.name}`, async ({ page }) => {
			await page.setViewportSize({ width: viewport.width, height: viewport.height });
			await page.route('**/api/setup/status', (route) =>
				fulfillJson(route, { needs_owner: true, registration_open: true, claim_requires_token: false })
			);
			await page.goto('/setup/claim');

			await expect(page.getByRole('heading', { name: 'A local studio for AI images and video' })).toBeVisible({
				timeout: 20000
			});
			await expect(page.getByText('Runs on your computer', { exact: true }).first()).toBeVisible();
			await expect(page.getByText('ComfyUI is optional')).toBeVisible();
			await expect(page.getByRole('button', { name: 'Skip the guide, go to the app' })).toHaveCount(0);
			await noHorizontalOverflow(page);
			await screenshot(page, JOURNEY, `welcome-${viewport.name}`);

			await page.getByRole('button', { name: 'Get started' }).click();
			await expect(page.getByRole('heading', { name: 'Create your owner account' })).toBeVisible();
			await expect(page.getByText('This account lives only on this computer.')).toBeVisible();
			await expect(page.getByText('storage/db.sqlite')).toBeVisible();
			await expect(page.getByText('Sign-up is closed once you exist.')).toBeVisible();
			await expect(page.getByText('Step 2 of 7 - Required')).toBeVisible();
						await expect(page.locator('label[for="email"]')).toHaveText(/^\s*Email\s*$/);
			await expect(page.getByRole('button', { name: 'Create owner account' })).toBeVisible();
			await expect(page.getByRole('button', { name: 'Skip guide' })).toHaveCount(0);
			await noHorizontalOverflow(page);
			await screenshot(page, JOURNEY, `account-${viewport.name}`);

			await page.getByRole('button', { name: 'Back', exact: true }).click();
			await expect(page.getByRole('heading', { name: 'A local studio for AI images and video' })).toBeVisible();
		});
	}

	test('a returning visitor lands on the account step', async ({ page }) => {
		await page.route('**/api/setup/status', (route) =>
			fulfillJson(route, { needs_owner: true, registration_open: true, claim_requires_token: false })
		);
		await page.goto('/setup/claim');
		await page.getByRole('button', { name: 'Get started' }).click();
		await page.reload();
		await expect(page.getByRole('heading', { name: 'Create your owner account' })).toBeVisible({ timeout: 20000 });
	});
});

test.describe('guide in the app shell', () => {
	test.use({ viewport: { width: 1440, height: 900 } });

	test('stepper progresses and the starter cards show size, source host and graphics memory', async ({ page }) => {
		await mockGuideApi(page);
		await page.goto('/setup');

		await expect(page.getByRole('heading', { name: 'Your computer' })).toBeVisible({ timeout: 20000 });
		await expect(page.locator('[aria-current="step"]')).toContainText('Computer');
		await expect(page.getByText('NVIDIA GeForce RTX 4090')).toBeVisible();
		await expect(page.getByText('· 24 GB')).toBeVisible();
		await expect(page.getByText('Good fit.')).toBeVisible();
		await expect(page.getByRole('button', { name: 'Back to Account' })).toHaveCount(0);
		await screenshot(page, JOURNEY, 'computer');

		await page.getByRole('button', { name: 'Continue' }).click();
		await expect(page.getByRole('heading', { name: 'Where should PotionUI keep your models?' })).toBeVisible();
		await expect(page.locator('[aria-current="step"]')).toContainText('Models');
		await expect(page.getByRole('button', { name: 'Back to Computer' })).toBeVisible();
		await page.getByRole('button', { name: 'Continue' }).click();

		await expect(page.getByRole('heading', { name: 'Pick a starter model' })).toBeVisible();
		await expect(page.locator('[aria-current="step"]')).toContainText('Starter');
		const cards = page.getByRole('radio');
		await expect(cards).toHaveCount(3);
		await expect(cards.nth(0)).toContainText('Z-Image Starter');
		await expect(cards.nth(0)).toContainText('Recommended for your card');
		await expect(cards.nth(0)).toHaveAttribute('aria-checked', 'true');
		await expect(cards.nth(0)).toContainText('19.27 GB');
		await expect(cards.nth(0)).toContainText('huggingface.co');
		await expect(cards.nth(0)).toContainText('12 to 16 GB suggested');
		await expect(cards.nth(0)).toContainText('Fits your card');
		await expect(cards.nth(1)).toContainText('SDXL Starter');
		await expect(cards.nth(1)).toContainText('6.62 GB');
		await expect(cards.nth(1)).toContainText('8 GB minimum');
		await expect(cards.nth(2)).toContainText('Qwen-Image Starter');
		await expect(cards.nth(2)).toContainText('28 GB');
		await expect(cards.nth(2)).toContainText('16 GB recommended');
		await expect(page.getByRole('button', { name: 'Review 19.27 GB download' })).toBeVisible();
		await screenshot(page, JOURNEY, 'starter');

		await cards.nth(1).click();
		await expect(page.getByRole('button', { name: 'Review 6.62 GB download' })).toBeVisible();
		await page.keyboard.press('ArrowDown');
		await expect(cards.nth(2)).toHaveAttribute('aria-checked', 'true');

		await page.getByRole('button', { name: 'Back to Models' }).click();
		await expect(page.getByRole('heading', { name: 'Where should PotionUI keep your models?' })).toBeVisible();
	});

	test('skip leads to the app and Resume setup in the sidebar leads back', async ({ page }) => {
		await mockGuideApi(page);
		await page.goto('/setup');
		await page.getByRole('button', { name: 'Continue' }).click();
		await page.getByRole('button', { name: 'Continue' }).click();
		await expect(page.getByRole('heading', { name: 'Pick a starter model' })).toBeVisible();

		await page.getByRole('button', { name: 'Skip the guide, go to the app' }).click();
		await page.waitForURL(/\/generate/);
		await expect(
			page.getByText('Guide skipped. Pick it up again any time from Resume setup.')
		).toBeVisible();
		await screenshot(page, JOURNEY, 'skipped-landing');

		const resume = page.getByRole('link', { name: 'Resume setup' });
		await expect(resume).toBeVisible({ timeout: 15000 });
		await resume.click();
		await page.waitForURL(/\/setup/);
		await expect(page.getByRole('heading', { name: 'Pick a starter model' })).toBeVisible();
	});

	test('the token disclosure on the consent card saves through the plugin settings route', async ({ page }) => {
		const consentRun = {
			id: 'run-1',
			recipe_id: 'zimage-starter',
			recipe_version: 1,
			scope: 'instance',
			mode: 'onboarding',
			status: 'awaiting_consent',
			current_step: 'plan',
			safe_input: null,
			safe_output: null,
			error_code: null,
			safe_error_detail: null,
			created_at: null,
			updated_at: null,
			completed_at: null,
			attempts: [],
			steps: [
				{
					step_key: 'plan',
					title: 'Plan required downloads',
					kind: 'artifacts.plan',
					ordinal: 0,
					status: 'awaiting_consent',
					attempts: [
						{
							step_key: 'plan',
							attempt: 1,
							status: 'awaiting_consent',
							progress_current: null,
							progress_total: null,
							progress_unit: null,
							safe_output: {
								consent_request: {
									artifacts: [
										{
											id: 'a1',
											display_name: 'Z-Image Turbo',
											size_bytes: 12 * GIB,
											kind: 'diffusion_model',
											gated: false,
											license_url: null
										}
									],
									total_bytes: 12 * GIB,
									providers: [
										{
											id: 'huggingface',
											name: 'Hugging Face',
											website: 'https://huggingface.co/settings/tokens',
											field_name: 'api_key',
											configured: false
										}
									]
								}
							},
							error_code: null,
							safe_error_detail: null,
							safe_suggested_action: null,
							started_at: null,
							finished_at: null
						}
					]
				}
			]
		};
		await mockGuideApi(page, consentRun);
		let saved: unknown = null;
		await page.route('**/api/plugins/huggingface-provider/settings', async (route) => {
			saved = route.request().postDataJSON();
			await fulfillJson(route, { data: [{ key: 'api_key', value: '***' }] });
		});

		await page.goto('/setup');
		await page.getByRole('button', { name: 'Continue' }).click();
		await page.getByRole('button', { name: 'Continue' }).click();
		await page.getByRole('button', { name: /^Review .+ download$/ }).click();

		await expect(page.getByRole('heading', { name: 'Review the download' })).toBeVisible();
		await expect(page.getByText('Image model')).toBeVisible();
		await expect(page.getByText(/Downloads come from Hugging Face \(huggingface\.co\)/)).toBeVisible();

		const disclosure = page.locator('details', { hasText: 'Add a token (optional)' });
		await expect(disclosure).not.toHaveJSProperty('open', true);
		await disclosure.locator('summary').click();
		await expect(disclosure.getByText('Stored on this computer.')).toBeVisible();
		await expect(disclosure.getByRole('button', { name: 'Save' })).toBeDisabled();
		await disclosure.locator('input[type="password"]').fill('hf_example_token');
		await disclosure.getByRole('button', { name: 'Save' }).click();

		await expect(page.getByText('Hugging Face API key saved.')).toBeVisible();
		expect(saved).toEqual({ settings: { api_key: 'hf_example_token' } });
		await screenshot(page, JOURNEY, 'consent-token-saved');
	});
});

test.describe('guide on a phone', () => {
	test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });

	test('skip promises Settings and Resume setup there leads back', async ({ page }) => {
		await mockGuideApi(page);
		await page.goto('/setup');

		await expect(page.getByText('Step 3 of 7').first()).toBeVisible({ timeout: 20000 });
		await noHorizontalOverflow(page);
		await screenshot(page, JOURNEY, 'computer-390');

		await page.getByRole('button', { name: 'Skip the guide, go to the app' }).click();
		await page.waitForURL(/\/generate/);
		await expect(
			page.getByText('Guide skipped. Pick it up again any time from Settings, under Resume setup.')
		).toBeVisible();

		await page.goto('/settings');
		const row = page.getByRole('button', { name: 'Resume setup' });
		await expect(row).toBeVisible({ timeout: 15000 });
		await screenshot(page, JOURNEY, 'settings-resume-390');
		await row.click();
		await page.waitForURL(/\/setup/);
		await expect(page.getByText('Step 3 of 7').first()).toBeVisible();
	});
});
