import { test, expect, type Page } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot } from './helpers';

const JOURNEY = 'model-tools-preset';
const PRESET_ID = '01M5ARBE0QD9PXX1XMNDXYVAK6';
const GB = 1024 ** 3;

const VIEWPORTS = [
	{ label: '1440', width: 1440, height: 900 },
	{ label: '390', width: 390, height: 844 }
] as const;

const MODES = [
	{
		key: 'quantize',
		label: 'Quantize',
		visible: ['Quantization Format', 'Output File Name', 'Overwrite Existing File'],
		absent: ['LoRAs', 'Method'],
		modelPicker: true
	},
	{
		key: 'merge_into_model',
		label: 'Merge into Model',
		visible: ['LoRAs', 'Output File Name', 'Overwrite Existing File'],
		absent: ['Method', 'Target Rank', 'Quantization Format'],
		modelPicker: true
	},
	{
		key: 'merge_and_quantize',
		label: 'Merge and Quantize',
		visible: ['LoRAs', 'Quantization Format', 'Output File Name', 'Overwrite Existing File'],
		absent: ['Method', 'Target Rank'],
		modelPicker: true
	},
	{
		key: 'merge_loras',
		label: 'Merge LoRAs',
		visible: ['LoRAs', 'Method', 'Target Rank', 'Output File Name', 'Overwrite Existing File'],
		absent: ['Quantization Format'],
		modelPicker: false
	}
] as const;

function modelRow(id: string, filename: string, modelType: string, sizeGb: number) {
	const now = new Date().toISOString();
	return {
		id,
		filename,
		model_type: modelType,
		file_size: Math.round(sizeGb * GB),
		sha256: id.padEnd(64, '0'),
		created_at: now,
		updated_at: now,
		indexed_at: now,
		is_available: true,
		is_favorite: false,
		tags: [],
		providers: [],
		files: [],
		backend_ids: ['native']
	};
}

const FORMAT_NOTES = [
	{ label: 'fp8 e4m3', note: 'RTX 40/50' },
	{ label: 'fp8 e5m2', note: 'no speedup here' },
	{ label: 'int8', note: 'not for text encoders' },
	{ label: 'int8 ConvRot', note: 'ComfyUI' },
	{ label: 'nvfp4', note: 'RTX 50' },
	{ label: 'int4 ConvRot', note: 'loads as int8 here' }
] as const;

const DIT_ROWS = [
	modelRow('e2e-dit-flux', 'flux-dev.safetensors', 'diffusion_model', 23.8),
	modelRow('e2e-dit-qwen', 'qwen-image-bf16.safetensors', 'diffusion_model', 40.9),
	modelRow('e2e-unet-wan', 'wan2.2-t2v-high.safetensors', 'unet', 28.6),
	modelRow('e2e-te-umt5', 'umt5-xxl-bf16.safetensors', 'text_encoder', 11.4)
];

const LORA_ROWS = [
	modelRow('e2e-lora-ink', 'ink-wash.safetensors', 'lora', 0.37),
	modelRow('e2e-lora-grain', 'paper-grain.safetensors', 'lora', 0.21)
];

interface ModelsRequest {
	modelType: string | null;
	modelTypes: string | null;
	unquantizedOnly: string | null;
}

async function mockPresetModels(page: Page): Promise<ModelsRequest[]> {
	const requests: ModelsRequest[] = [];
	await page.route(`**/api/presets/${PRESET_ID}/models**`, async (route) => {
		const url = new URL(route.request().url());
		const modelType = url.searchParams.get('model_type');
		requests.push({
			modelType,
			modelTypes: url.searchParams.get('model_types'),
			unquantizedOnly: url.searchParams.get('unquantized_only')
		});
		const models = modelType === 'lora' ? LORA_ROWS : DIT_ROWS;
		await route.fulfill({
			json: { success: true, data: { engine: 'native', models, total: models.length, indexed: true } }
		});
	});
	return requests;
}

async function installAndOpen(page: Page): Promise<boolean> {
	const token = await ownerToken(page);
	const headers = { Authorization: `Bearer ${token}` };
	const list = await page.request.get('/api/presets?include_uninstalled=true', { headers });
	const presets = ((await list.json().catch(() => ({}))).data || []) as Array<{
		id: string;
		name: string;
		installed?: boolean;
	}>;
	const preset = presets.find((p) => p.id === PRESET_ID);
	if (!preset) return false;
	if (!preset.installed) {
		const install = await page.request.post(`/api/presets/${PRESET_ID}/install`, { headers });
		expect(install.ok(), `install -> ${install.status()}`).toBeTruthy();
	}
	const me = await page.request.get('/api/auth/me', { headers });
	const userId = (await me.json())?.data?.id as string;
	const assign = await page.request.post(`/api/presets/${PRESET_ID}/assign`, {
		headers,
		data: { user_ids: [userId] }
	});
	expect(assign.ok(), `assign -> ${assign.status()}`).toBeTruthy();

	await page.goto('/generate');
	await page.waitForLoadState('networkidle');
	const sheet = await openPresetSurface(page);
	await sheet
		.locator('button[aria-haspopup="dialog"]', isPhone(page) ? {} : { hasText: 'Choose a preset' })
		.first()
		.click();
	await page.getByText(preset.name, { exact: true }).first().click();
	await page.getByRole('button', { name: /Use this preset|Keep selected/ }).click();
	await closePresetSurface(page);
	return true;
}

function isPhone(page: Page): boolean {
	return (page.viewportSize()?.width ?? 1440) < 768;
}

async function openPresetSurface(page: Page) {
	if (!isPhone(page)) return page.locator('body');
	await page.getByRole('button', { name: 'Open preset and session' }).click();
	const sheet = page.getByRole('dialog', { name: 'Preset and session' });
	await expect(sheet).toBeVisible();
	return sheet;
}

async function closePresetSurface(page: Page) {
	if (!isPhone(page)) return;
	const sheet = page.getByRole('dialog', { name: 'Preset and session' });
	if (await sheet.isVisible()) await page.keyboard.press('Escape');
	await expect(sheet).toBeHidden();
}

async function chooseMode(page: Page, label: string) {
	const surface = await openPresetSurface(page);
	const modeSelect = surface.getByTestId('preset-header-mode');
	await expect(modeSelect).toBeVisible({ timeout: 20000 });
	await modeSelect.getByRole('button').first().click();
	await page.getByRole('option', { name: new RegExp(`^${label}\\b`) }).first().click();
	await closePresetSurface(page);
	if (isPhone(page)) {
		const settings = page.getByRole('dialog', { name: 'Settings' });
		if (!(await settings.isVisible())) await page.getByRole('button', { name: 'Settings' }).first().click();
		await expect(settings).toBeVisible();
	}
}

function fieldLabel(page: Page, label: string) {
	return page.getByText(label, { exact: true }).first();
}

test.afterEach(async ({ page }) => {
	await page.unrouteAll({ behavior: 'ignoreErrors' });
});

for (const viewport of VIEWPORTS) {
	test(`every mode shows its form at ${viewport.label}`, async ({ page }) => {
		test.setTimeout(120000);
		await page.setViewportSize({ width: viewport.width, height: viewport.height });
		await loginAsOwner(page);
		await mockPresetModels(page);
		if (!(await installAndOpen(page))) {
			test.skip(true, 'Model Tools preset is not available on this throwaway instance.');
			return;
		}

		for (const mode of MODES) {
			await chooseMode(page, mode.label);
			await expect(fieldLabel(page, 'Output File Name')).toBeVisible({ timeout: 20000 });
			for (const label of mode.visible) {
				await expect(fieldLabel(page, label), `${mode.key}: ${label}`).toBeVisible();
			}
			for (const label of mode.absent) {
				await expect(fieldLabel(page, label), `${mode.key}: no ${label}`).toHaveCount(0);
			}
			await expect(page.getByPlaceholder('Select a model...')).toHaveCount(mode.modelPicker ? 1 : 0);
			await screenshot(page, JOURNEY, `${mode.key}-${viewport.label}`);
			if (isPhone(page)) await page.keyboard.press('Escape');
		}
	});
}

test('the model picker lists DiT and text-encoder files together and asks for unquantized ones in Quantize', async ({
	page
}) => {
	test.setTimeout(120000);
	await page.setViewportSize({ width: 1440, height: 900 });
	await loginAsOwner(page);
	const requests = await mockPresetModels(page);
	if (!(await installAndOpen(page))) {
		test.skip(true, 'Model Tools preset is not available on this throwaway instance.');
		return;
	}

	await chooseMode(page, 'Quantize');
	await page.getByPlaceholder('Select a model...').click();
	for (const row of DIT_ROWS) {
		await expect(page.getByText(row.filename.replace('.safetensors', ''), { exact: false }).first()).toBeVisible({
			timeout: 15000
		});
	}
	const quantizeRequest = requests.find((r) => r.modelTypes?.includes('text_encoder'));
	expect(quantizeRequest, JSON.stringify(requests)).toBeTruthy();
	expect(quantizeRequest?.modelTypes).toBe('diffusion_model,unet,checkpoint,text_encoder');
	expect(quantizeRequest?.unquantizedOnly).toBe('true');
	await screenshot(page, JOURNEY, 'quantize-model-picker-1440');
	await page.keyboard.press('Escape');

	requests.length = 0;
	await chooseMode(page, 'Merge into Model');
	await page.getByPlaceholder('Select a model...').click();
	await expect(page.getByText('flux-dev', { exact: false }).first()).toBeVisible({ timeout: 15000 });
	const mergeRequest = requests.find((r) => r.modelTypes);
	expect(mergeRequest?.modelTypes).toBe('diffusion_model,unet,checkpoint');
	expect(mergeRequest?.unquantizedOnly).toBeNull();
	await page.keyboard.press('Escape');

	requests.length = 0;
	await page.getByRole('button', { name: /Add LoRA/ }).first().click();
	await page.getByPlaceholder('Select a LoRA...').click();
	await expect(page.getByText('ink-wash', { exact: false }).first()).toBeVisible({ timeout: 15000 });
	const loraRequest = requests.find((r) => r.modelType === 'lora');
	expect(loraRequest?.modelTypes).toBeNull();
	expect(loraRequest?.unquantizedOnly).toBeNull();
	await screenshot(page, JOURNEY, 'merge-lora-picker-1440');
});

test('the file name field refuses path separators before anything is sent', async ({ page }) => {
	test.setTimeout(120000);
	await page.setViewportSize({ width: 1440, height: 900 });
	await loginAsOwner(page);
	await mockPresetModels(page);
	if (!(await installAndOpen(page))) {
		test.skip(true, 'Model Tools preset is not available on this throwaway instance.');
		return;
	}

	await chooseMode(page, 'Quantize');
	const input = page.locator('#output_name');
	await expect(input).toBeVisible({ timeout: 20000 });
	await input.fill('../escape/me');
	await expect(page.getByText(/cannot contain/)).toBeVisible();
	await screenshot(page, JOURNEY, 'quantize-bad-name-1440');
	await input.fill('flux-dev-fp8');
	await expect(page.getByText(/cannot contain/)).toHaveCount(0);
});

test('the quantize modes offer every format with a note on what it gives on this hardware', async ({ page }) => {
	test.setTimeout(120000);
	await page.setViewportSize({ width: 1440, height: 900 });
	await loginAsOwner(page);
	await mockPresetModels(page);
	if (!(await installAndOpen(page))) {
		test.skip(true, 'Model Tools preset is not available on this throwaway instance.');
		return;
	}

	for (const mode of ['Quantize', 'Merge and Quantize']) {
		await chooseMode(page, mode);
		const picker = page.locator('[aria-labelledby="quant_format-label"]').locator('button[aria-haspopup="listbox"]');
		await expect(picker).toContainText('fp8 e4m3', { timeout: 20000 });
		await picker.click();
		const options = page.getByRole('option');
		await expect(options).toHaveCount(FORMAT_NOTES.length);
		for (const [index, format] of FORMAT_NOTES.entries()) {
			await expect(options.nth(index)).toContainText(format.label);
			await expect(options.nth(index)).toContainText(format.note);
		}
		await screenshot(page, JOURNEY, `${mode.toLowerCase().replace(/ /g, '-')}-formats-1440`);
		await options.filter({ hasText: 'nvfp4' }).first().click();
		await expect(picker).toContainText('nvfp4');
	}
});
