import { test, expect, type WebSocketRoute } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';
import { installAndSelectImagePreset } from './presetPreamble';
import { makeModelFieldsOptional, generationStatusMessage } from './generationMocks';

const JOURNEY = 'generation-stage-text';
const GENERATION_ID = '01STAGETEXT0000000000001';
const MODEL = 'e2e-checkpoint.safetensors';
const VIEWPORTS = [
	{ label: '1440', width: 1440, height: 900 },
	{ label: '390', width: 390, height: 844 }
];
const ROLES = [
	{ label: 'admin', admin: true },
	{ label: 'user', admin: false }
];

for (const role of ROLES) {
	for (const viewport of VIEWPORTS) {
		test(`generation stage text — ${role.label} at ${viewport.label}`, async ({ page }) => {
			await page.setViewportSize({ width: 1440, height: 900 });
			const sockets: WebSocketRoute[] = [];
			const send = (ws: WebSocketRoute, message: object) => ws.send(JSON.stringify(message));

			await page.routeWebSocket(/\/ws\/generation/, (ws) => {
				sockets.push(ws);
				send(ws, { type: 'connection_established', client_id: 'e2e-mock' });
				ws.onMessage((raw) => {
					const msg = JSON.parse(raw as string);
					if (msg.type === 'subscribe_generation') send(ws, { type: 'subscribed', generation_id: msg.generation_id });
				});
			});

			await page.route('**/api/generations/start', async (route) => {
				await route.fulfill({
					json: {
						success: true,
						data: {
							generation_id: GENERATION_ID,
							status: { id: GENERATION_ID, generation_id: GENERATION_ID, status: 'pending', progress: 0.0, created_at: new Date().toISOString() },
							queue_position: null
						}
					}
				});
			});

			if (!role.admin) {
				await page.route('**/api/auth/me', async (route) => {
					const response = await route.fetch();
					const body = await response.json();
					if (body?.data) body.data.account_type = 'USER';
					await route.fulfill({ response, json: body });
				});
			}

			await makeModelFieldsOptional(page);
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
			await page.setViewportSize({ width: viewport.width, height: viewport.height });

			send(ws, generationStatusMessage({
				generationId: GENERATION_ID,
				pipeId: 1,
				pipeName: 'checkpoint_loader/sdxl',
				currentStep: `Loading checkpoint <<MODEL:${MODEL}>> <<PROGRESS:3/30>>`,
				progress: 0.1,
				stepNum: 3,
				totalSteps: 30
			}));

			const detail = page.locator('.deck-detail-text');
			await expect(detail).toContainText('Loading model', { timeout: 10000 });
			await expect(detail).toContainText('Step 3 / 30');
			await expect(detail).not.toContainText('<<');
			await expect(detail).not.toContainText('checkpoint_loader');

			const adminMark = page.locator('[data-admin-stage-detail]');
			if (role.admin) {
				await expect(adminMark).toBeVisible();
			} else {
				await expect(adminMark).toHaveCount(0);
				await expect(detail).not.toContainText(MODEL);
			}

			send(ws, generationStatusMessage({
				generationId: GENERATION_ID,
				pipeId: 2,
				pipeName: 'acme_internal_pipe',
				currentStep: 'Sampling <<PROGRESS:21/30>>',
				progress: 0.7,
				stepNum: 21,
				totalSteps: 30
			}));
			await expect(detail).toContainText('Working…', { timeout: 10000 });
			await expect(detail).not.toContainText('acme_internal_pipe');

			send(ws, {
				...generationStatusMessage({
					generationId: GENERATION_ID,
					pipeId: 1,
					pipeName: 'generator',
					currentStep: 'Sampling <<PROGRESS:5/30>>',
					progress: 0.4,
					stepNum: 5,
					totalSteps: 30
				}),
				prompt_index: 1,
				prompt_count: 3,
				item_index: 1,
				item_count: 3,
				item_progress: 0.17
			});
			await expect(page.locator('[data-batch-progress]')).toHaveText('Prompt 2 of 3', { timeout: 10000 });
			await screenshot(page, JOURNEY, `${role.label}-${viewport.label}`);
		});
	}
}
