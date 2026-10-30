import { test, expect, type Page, type WebSocketRoute } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';
import { makeModelFieldsOptional, generationStatusMessage } from './generationMocks';
import { discoverAndAssignImagePreset, presetOption } from './presetPreamble';

const JOURNEY = 'mobile-generation-deck';
const GENERATION_ID = '01MOBILEDECK0000000000001';
const TOTAL_STEPS = 30;
const PREVIEW_IMAGE =
	'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';

test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });

const statusMsg = (stepNum: number) =>
	generationStatusMessage({
		generationId: GENERATION_ID,
		pipeId: 1,
		pipeName: 'generator',
		currentStep: `Sampling <<PROGRESS:${stepNum}/${TOTAL_STEPS}>>`,
		progress: stepNum / TOTAL_STEPS,
		stepNum,
		totalSteps: TOTAL_STEPS
	});

async function selectPresetOnPhone(page: Page): Promise<boolean> {
	const preset = await discoverAndAssignImagePreset(page);
	if (!preset) return false;

	await page.goto('/generate');
	await expect(page.locator('.studio-dock')).toBeVisible({ timeout: 15000 });
	await page.getByRole('button', { name: 'Open preset and session' }).click();
	const sheet = page.getByRole('dialog', { name: 'Preset and session' });
	await expect(sheet).toBeVisible({ timeout: 5000 });
	await sheet.locator('button[aria-haspopup="dialog"]', { hasText: 'Choose a preset' }).click();
	await expect(page.locator('[role="listbox"][aria-label="Presets"]')).toBeVisible({ timeout: 15000 });
	await presetOption(page, preset.name).click();
	await expect(page.getByRole('dialog', { name: 'Choose preset' })).toBeHidden();
	await page.keyboard.press('Escape');
	await expect(sheet).toHaveCount(0);
	return true;
}

async function typePrompt(page: Page, text: string) {
	await page.getByRole('button', { name: 'Prompt', exact: true }).click();
	const editor = page.locator('[contenteditable="true"]').first();
	await expect(editor).toBeVisible({ timeout: 8000 });
	await editor.click();
	await page.keyboard.type(text);
	await page.keyboard.press('Escape');
	await expect(page.getByRole('dialog', { name: 'Prompt', exact: true })).toHaveCount(0);
}

test('the phone deck shows one loader, a literal status and Cancel while a run is live', async ({ page }) => {
	const sockets: WebSocketRoute[] = [];
	const subscribed = new Set<string>();
	const send = (ws: WebSocketRoute, message: object) => ws.send(JSON.stringify(message));

	await page.routeWebSocket(/\/ws\/generation/, (ws) => {
		sockets.push(ws);
		send(ws, { type: 'connection_established', client_id: 'e2e-mock' });
		ws.onMessage((raw) => {
			const msg = JSON.parse(raw as string);
			if (msg.type === 'subscribe_generation') {
				subscribed.add(msg.generation_id);
				send(ws, { type: 'subscribed', generation_id: msg.generation_id });
			}
		});
	});
	await page.route('**/api/generations/start', async (route) => {
		await route.fulfill({
			json: {
				success: true,
				data: {
					generation_id: GENERATION_ID,
					status: {
						id: GENERATION_ID,
						generation_id: GENERATION_ID,
						status: 'pending',
						progress: 0.0,
						created_at: new Date().toISOString()
					},
					queue_position: null
				}
			}
		});
	});
	await makeModelFieldsOptional(page);

	await loginAsOwner(page);
	if (!(await selectPresetOnPhone(page))) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return;
	}
	await typePrompt(page, 'a glowing potion bottle on a wooden alchemist bench');

	const disc = page.locator('.generation-panel .generate-button');
	await expect(disc).toBeEnabled({ timeout: 15000 });
	await expect(disc).toHaveAttribute('aria-label', 'Generate');
	await disc.click();

	await expect.poll(() => sockets.length, { timeout: 15000 }).toBeGreaterThan(0);
	await expect.poll(() => subscribed.has(GENERATION_ID), { timeout: 15000 }).toBe(true);
	const ws = sockets[sockets.length - 1];
	send(ws, statusMsg(9));

	const bar = page.getByRole('progressbar', { name: 'Generation progress' });
	await expect(page.locator('[role="progressbar"]')).toHaveCount(1);
	await expect(bar).toBeVisible({ timeout: 10000 });
	await expect(bar).toHaveAttribute('aria-valuenow', '30', { timeout: 10000 });

	const deckBox = await page.locator('.generation-panel').boundingBox();
	const barBox = await bar.boundingBox();
	expect(Math.abs(barBox!.width - deckBox!.width), 'loader spans the deck').toBeLessThanOrEqual(1);
	expect(Math.abs(barBox!.height - 3), 'loader is 3px').toBeLessThanOrEqual(1);
	expect(Math.abs(barBox!.y - deckBox!.y), 'loader sits on the top edge').toBeLessThanOrEqual(2);

	await expect(page.locator('.studio-dock svg circle')).toHaveCount(0);
	await expect(disc).toHaveAttribute('aria-label', 'Cancel generation');
	await expect(page.locator('[data-deck-literal]')).toHaveText('Generating');
	await expect(page.locator('[data-deck-title]')).toHaveAttribute('aria-hidden', 'true');
	await expect(page.locator('.deck-detail-text')).toContainText('30%');
	await expect(page.locator('.deck-detail-text')).toContainText('Step 9 / 30');
	await expect(page.locator('.deck-eyebrow')).toBeHidden();

	await page.getByRole('button', { name: 'More generation actions' }).click();
	await expect(page.locator('[data-deck-more-elapsed]')).toContainText('Elapsed');
	await screenshot(page, JOURNEY, 'running-more-open');
	await page.keyboard.press('Escape');
	await expect(page.locator('[data-deck-more-elapsed]')).toHaveCount(0);

	await screenshot(page, JOURNEY, 'running');

	send(ws, {
		type: 'workbench_update',
		generation_id: GENERATION_ID,
		pipe_id: 1,
		pipe_name: 'generator',
		output_type: 'image',
		temporary: true,
		seed: 1,
		resolution: [1024, 1024],
		step: TOTAL_STEPS,
		image: PREVIEW_IMAGE
	});
	send(ws, {
		type: 'gallery_update',
		generation_id: GENERATION_ID,
		pipe_id: 2,
		pipe_name: 'gallery',
		output_type: 'gallery',
		images: [PREVIEW_IMAGE],
		image_urls_list: [{ derived: false, seed: 1, resolution: '1024x1024', step: TOTAL_STEPS }]
	});
	send(ws, {
		type: 'generation_complete',
		data: {
			id: GENERATION_ID,
			generation_id: GENERATION_ID,
			status: 'completed',
			progress: 1.0,
			created_at: new Date().toISOString(),
			completed_at: new Date().toISOString()
		}
	});

	await expect(page.locator('[data-deck-literal]')).toHaveText('Done', { timeout: 10000 });
	await expect(bar).toBeHidden();
	await expect(disc).toHaveAttribute('aria-label', 'Generate');
	await expect(page.locator('[data-studio-result-row]')).toBeVisible();
	await screenshot(page, JOURNEY, 'done');
});

test('Continuous arms from the phone deck, locks X/Y out and starts nothing by itself', async ({ page }) => {
	const starts: string[] = [];
	page.on('request', (request) => {
		if (request.method() === 'POST' && request.url().includes('/api/generations/start')) starts.push(request.url());
	});
	await makeModelFieldsOptional(page);

	await loginAsOwner(page);
	if (!(await selectPresetOnPhone(page))) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return;
	}
	await typePrompt(page, 'a glowing potion bottle on a wooden alchemist bench');

	const controls = page.locator('.generation-panel .run-cluster button');
	const continuous = controls.nth(2);
	const compare = page.locator('.generation-panel [data-compare-toggle]');
	await expect(page.locator('.generation-panel .generate-button')).toBeEnabled({ timeout: 15000 });

	await expect(continuous).toHaveAttribute('aria-pressed', 'false');
	await expect(continuous).toHaveAttribute('aria-label', 'Mode: Generate once');
	await continuous.click();
	await expect(continuous).toHaveAttribute('aria-pressed', 'true');
	await expect(continuous).toHaveAttribute('aria-label', 'Mode: Continuous');
	await expect(compare).toBeDisabled();
	await expect(page.locator('.deck-detail-text')).toContainText('Continuous mode armed');
	await screenshot(page, JOURNEY, 'continuous-armed');

	await continuous.click();
	await expect(continuous).toHaveAttribute('aria-pressed', 'false');
	await expect(compare).toBeEnabled();
	expect(starts, 'arming Continuous must not start a generation').toHaveLength(0);
});
