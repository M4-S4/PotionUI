import { test, expect, type WebSocketRoute } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';
import { installAndSelectImagePreset } from './presetPreamble';

const JOURNEY = 'live-generation-progress';
const GENERATION_ID = '01LIVEGENPROGRESS00000001';
const MODEL = 'e2e-checkpoint.safetensors';
const TOTAL_STEPS = 30;
const SEED = 74213;
const PREVIEW_IMAGE =
	'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';

function statusMsg(pipeId: number, pipeName: string, currentStep: string, progress: number, stepNum?: number) {
	return {
		type: 'generation_status',
		generation_id: GENERATION_ID,
		pipe_id: pipeId,
		pipe_name: pipeName,
		output_type: 'progress',
		status: 'running',
		current_step: currentStep,
		message: `<<PIPE:${pipeName}>>`,
		progress,
		...(stepNum !== undefined ? { current_step_num: stepNum, total_steps: TOTAL_STEPS } : {})
	};
}

function workbenchMsg(stepNum: number) {
	return {
		type: 'workbench_update',
		generation_id: GENERATION_ID,
		pipe_id: 1,
		pipe_name: 'generator',
		output_type: 'image',
		temporary: true,
		seed: SEED,
		resolution: [1024, 1536],
		sampler: 'DPMPP_2M',
		cfg: 6.5,
		step: stepNum,
		image: PREVIEW_IMAGE
	};
}

test('live generation progress — bar advances, preview refines, gallery strip appears', async ({ page }) => {
	const sockets: WebSocketRoute[] = [];
	const send = (ws: WebSocketRoute, message: object) => ws.send(JSON.stringify(message));

	await page.routeWebSocket(/\/ws\/generation/, (ws) => {
		sockets.push(ws);
		send(ws, { type: 'connection_established', client_id: 'e2e-mock' });
		ws.onMessage((raw) => {
			const msg = JSON.parse(raw as string);
			if (msg.type === 'subscribe_generation') {
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

	await loginAsOwner(page);
	const preset = await installAndSelectImagePreset(page);
	if (!preset) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return;
	}

	const promptEditor = page.locator('.inline-chip-editor[role="textbox"]').first();
	await expect(promptEditor).toBeVisible({ timeout: 15000 });
	await promptEditor.click();
	await page.keyboard.type('a glowing potion bottle on a wooden alchemist bench');

	const generateButton = page.getByRole('button', { name: 'Generate', exact: true });
	await expect(generateButton).toBeEnabled({ timeout: 15000 });
	await generateButton.click();

	await expect.poll(() => sockets.length, { timeout: 15000 }).toBeGreaterThan(0);
	const ws = sockets[sockets.length - 1];

	send(ws, statusMsg(0, 'model_loader', `Loading checkpoint <<MODEL:${MODEL}>>`, 0.0));
	send(ws, statusMsg(1, 'generator', 'Sampling <<PROGRESS:3/30>>', 3 / TOTAL_STEPS, 3));

	const progressBar = page.getByRole('progressbar', { name: 'Generation progress' });
	await expect(progressBar).toBeVisible({ timeout: 10000 });
	await expect(progressBar).toHaveAttribute('aria-valuenow', '10', { timeout: 10000 });

	send(ws, workbenchMsg(3));
	const preview = page.locator('img[src^="data:image/png;base64"]').first();
	await expect(preview).toBeVisible({ timeout: 10000 });

	send(ws, statusMsg(1, 'generator', 'Sampling <<PROGRESS:21/30>>', 21 / TOTAL_STEPS, 21));
	await expect(progressBar).toHaveAttribute('aria-valuenow', '70', { timeout: 10000 });
	await screenshot(page, JOURNEY, '01-sampling');

	send(ws, statusMsg(1, 'generator', 'Sampling <<PROGRESS:30/30>>', 1.0, 30));
	await expect(progressBar).toHaveAttribute('aria-valuenow', '100', { timeout: 10000 });

	send(ws, {
		type: 'gallery_update',
		generation_id: GENERATION_ID,
		pipe_id: 2,
		pipe_name: 'gallery',
		output_type: 'gallery',
		images: [PREVIEW_IMAGE, PREVIEW_IMAGE],
		image_urls_list: [
			{ derived: false, seed: SEED, sampler: 'DPMPP_2M', cfg: 6.5, resolution: '1024x1536', step: TOTAL_STEPS },
			{ derived: false, seed: SEED + 1, sampler: 'DPMPP_2M', cfg: 6.5, resolution: '1024x1536', step: TOTAL_STEPS }
		]
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

	await expect(page.getByAltText(/Generated image/).first()).toBeVisible({ timeout: 10000 });
	await screenshot(page, JOURNEY, '02-gallery');
});
