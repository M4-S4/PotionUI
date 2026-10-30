import { test, expect } from '@playwright/test';
import { loginAsOwner, ownerToken, scopedStorageKey, screenshot } from './helpers';

const TABS_STORAGE_KEY = 'potionui_tabs_state';

const JOURNEY = 'model-file-card';
const WORKBENCH_ID = '01MODELFILECARDWORKBENCH01';
const GB = 1024 ** 3;

const VIEWPORTS = [
	{ label: '1440', width: 1440, height: 900 },
	{ label: '390', width: 390, height: 844 }
] as const;

const MODEL_DATA = {
	index: 0,
	kind: 'model',
	model_type: 'diffusion_model',
	file_name: 'flux-dev-fp8-scaled.safetensors',
	size_bytes: Math.round(11.9 * GB),
	source_size_bytes: Math.round(23.8 * GB),
	format: 'fp8_scaled',
	model_id: 'e2e-model-id',
	indexed: true,
	index_warning: null,
	summary: {
		layers_quantized: 412,
		layers_kept: 38,
		energy_kept: 0.987,
		worst_layers: [
			{ key: 'double_blocks.3.img_attn.qkv.weight', rel_l2: 0.0312, cosine: 0.9994 },
			{ key: 'single_blocks.17.linear2.weight', rel_l2: 0.0287, cosine: 0.9996 }
		],
		sources: [],
		licence_note: 'Derived from FLUX.1 [dev], non-commercial licence applies.'
	}
};

const LORA_DATA = {
	index: 1,
	kind: 'lora',
	model_type: 'lora',
	file_name: 'baked-style-merge.safetensors',
	size_bytes: Math.round(0.37 * GB),
	source_size_bytes: null,
	format: 'lora',
	model_id: 'e2e-lora-id',
	indexed: true,
	index_warning: null,
	summary: {
		rank: 32,
		energy_kept: 0.93,
		sources: [
			{ name: 'ink-wash.safetensors', sha256: 'a1b2c3d4'.repeat(8), strength: 0.8 },
			{ name: 'paper-grain.safetensors', sha256: 'e5f6a7b8'.repeat(8), strength: 0.45 }
		],
		licence_note: 'Merged from two community LoRAs; check each source licence.'
	}
};

function artifactMessage(generationId: string, data: typeof MODEL_DATA | typeof LORA_DATA) {
	return {
		type: 'pipe_artifact',
		generation_id: generationId,
		pipe_id: 3,
		pipe_name: 'model_tool',
		output_type: 'model_file',
		index: data.index,
		artifact_type: 'model_file',
		artifact_data: data
	};
}

for (const viewport of VIEWPORTS) {
	test(`model and LoRA cards in the Workbench at ${viewport.label}`, async ({ page }) => {
		await page.setViewportSize({ width: viewport.width, height: viewport.height });
		await loginAsOwner(page);
		const token = await ownerToken(page);
		const auth = { headers: { Authorization: `Bearer ${token}` } };
		const me = await (await page.request.get('/api/auth/me', auth)).json();
		const list = await (await page.request.get('/api/presets?include_uninstalled=true', auth)).json();
		const preset = ((list.data || []) as Array<{ id: string; installed?: boolean }>)[0];
		if (!preset) {
			test.skip(true, 'No presets available on this throwaway instance.');
			return;
		}
		if (!preset.installed) await page.request.post(`/api/presets/${preset.id}/install`, { ...auth, data: {} });
		await page.request.post(`/api/presets/${preset.id}/assign`, { ...auth, data: { user_ids: [me.data.id] } });
		const modes = await (await page.request.get(`/api/presets/${preset.id}/modes`, auth)).json();
		const modeName = modes.data?.modes?.[0]?.name as string | undefined;
		if (!modeName) {
			test.skip(true, 'Preset exposes no modes on this throwaway instance.');
			return;
		}

		await page.route(`**/api/generations/${WORKBENCH_ID}/status`, (route) =>
			route.fulfill({
				json: {
					success: true,
					data: {
						id: WORKBENCH_ID,
						generation_id: WORKBENCH_ID,
						status: 'running',
						progress: 0.4,
						created_at: new Date().toISOString()
					}
				}
			})
		);

		const channel: { send: ((message: object) => void) | null; subscribed: boolean } = {
			send: null,
			subscribed: false
		};
		await page.routeWebSocket('**/ws/generation*', (ws) => {
			channel.send = (message) => ws.send(JSON.stringify(message));
			ws.send(JSON.stringify({ type: 'connection_established' }));
			ws.onMessage((raw) => {
				const msg = JSON.parse(String(raw));
				if (msg.type === 'subscribe_generation') {
					ws.send(JSON.stringify({ type: 'subscribed', generation_id: msg.generation_id }));
					if (msg.generation_id === WORKBENCH_ID) channel.subscribed = true;
				}
				if (msg.type === 'ping') ws.send(JSON.stringify({ type: 'pong' }));
			});
		});

		await page.addInitScript(
			({ key, genId, presetId, mode }) => {
				const raw = localStorage.getItem(key);
				if (!raw) return;
				const state = JSON.parse(raw);
				if (!state?.tabs?.length) return;
				state.tabs[0].activeGenerationId = genId;
				state.tabs[0].selectedPreset = presetId;
				state.tabs[0].selectedMode = mode;
				localStorage.setItem(key, JSON.stringify(state));
			},
			{
				key: await scopedStorageKey(page, TABS_STORAGE_KEY),
				genId: WORKBENCH_ID,
				presetId: preset.id,
				mode: modeName
			}
		);
		await page.reload();
		await page.waitForURL(/\/generate/, { timeout: 15000 });
		await expect.poll(() => channel.send !== null && channel.subscribed, { timeout: 20000 }).toBe(true);

		channel.send!(artifactMessage(WORKBENCH_ID, MODEL_DATA));
		channel.send!(artifactMessage(WORKBENCH_ID, LORA_DATA));

		const cards = page.locator('[data-model-file-card]');
		await expect(cards).toHaveCount(2, { timeout: 10000 });
		await expect(page.locator('[data-model-file-sizes]').first()).toHaveText('23.8 → 11.9 GB');
		await expect(cards.nth(1)).toHaveAttribute('data-kind', 'lora');
		await expect(page.locator('[data-model-file-rank]')).toHaveText('32');

		await expect(page.getByText('Your creations appear here')).toHaveCount(0);
		await cards.first().scrollIntoViewIfNeeded();
		await page.getByRole('button', { name: /Worst layers/ }).first().click();
		await expect(page.locator('[data-model-file-worst]')).toBeVisible();

		const lastOpen = cards.last().getByRole('button', { name: 'Open in Models' });
		const stage = page.getByTestId('studio-model-file-stage');
		if (await stage.count()) {
			await stage.evaluate((el) => (el.scrollTop = el.scrollHeight));
		} else {
			await lastOpen.scrollIntoViewIfNeeded();
		}
		const buttonBox = await lastOpen.boundingBox();
		const dock = page.locator('.studio-dock');
		if (await dock.count()) {
			const dockBox = await dock.boundingBox();
			expect(buttonBox!.y + buttonBox!.height).toBeLessThanOrEqual(dockBox!.y);
		} else {
			const viewportHeight = page.viewportSize()!.height;
			expect(buttonBox!.y + buttonBox!.height).toBeLessThanOrEqual(viewportHeight);
		}
		await cards.first().scrollIntoViewIfNeeded();
		await expect(cards.first().locator('[data-model-file-name]')).toContainText('.safetensors');

		const overflow = await page.evaluate(
			() => document.documentElement.scrollWidth - document.documentElement.clientWidth
		);
		expect(overflow).toBeLessThanOrEqual(0);

		await screenshot(page, JOURNEY, `workbench-${viewport.label}`);
	});
}
