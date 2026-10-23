import { expect, type Locator, type Page } from '@playwright/test';
import { ownerToken, scopedStorageKey } from './helpers';

export interface DiscoveredPreset {
	id: string;
	name: string;
}

export async function discoverAndAssignImagePreset(page: Page): Promise<DiscoveredPreset | null> {
	const token = await ownerToken(page);
	const headers = { Authorization: `Bearer ${token}` };

	const list = await page.request.get('/api/presets?include_uninstalled=true', { headers });
	const listJson = await list.json().catch(() => ({}));
	const presets = (listJson.data || []) as Array<{
		id: string;
		name: string;
		engine?: string;
		category?: string;
		installed?: boolean;
	}>;
	const preset =
		presets.find((p) => /sdxl/i.test(p.id)) ||
		presets.find((p) => p.engine === 'native' && p.category === 'image');
	if (!preset) return null;

	if (!preset.installed) {
		const install = await page.request.post(`/api/presets/${preset.id}/install`, { headers });
		const installJson = await install.json().catch(() => ({}));
		expect(
			install.ok() && installJson.success,
			`preset install must succeed to render a prompt section: ${install.status()} ${JSON.stringify(installJson)}`
		).toBeTruthy();
	}

	const me = await page.request.get('/api/auth/me', { headers });
	const userId = (await me.json())?.data?.id as string;
	expect(userId, 'owner user id').toBeTruthy();
	const assign = await page.request.post(`/api/presets/${preset.id}/assign`, {
		headers,
		data: { user_ids: [userId] }
	});
	const assignJson = await assign.json().catch(() => ({}));
	expect(
		assign.ok() && assignJson.success,
		`preset assign must succeed: ${assign.status()} ${JSON.stringify(assignJson)}`
	).toBeTruthy();

	return { id: preset.id, name: preset.name };
}

export async function installAndSelectImagePreset(page: Page): Promise<DiscoveredPreset | null> {
	const preset = await discoverAndAssignImagePreset(page);
	if (!preset) return null;

	await page.goto('/generate');
	await page.waitForLoadState('networkidle');

	const pickerTrigger = page.locator('button[aria-haspopup="dialog"]', { hasText: 'Choose a preset' });
	await expect(pickerTrigger).toBeVisible({ timeout: 15000 });
	await pickerTrigger.click();
	const presetList = page.locator('[role="listbox"][aria-label="Presets"]');
	await expect(presetList).toBeVisible({ timeout: 15000 });
	await presetOption(page, preset.name).click();
	await expect(page.getByRole('dialog', { name: 'Choose preset' })).toBeHidden();
	await expectPresetPersisted(page, preset.id);

	return preset;
}

export function presetOption(page: Page, name: string): Locator {
	return page.getByRole('listbox', { name: 'Presets' }).locator('[data-preset-family]').getByRole('option', { name, exact: true });
}

export async function expectPresetPersisted(page: Page, presetId: string): Promise<void> {
	const tabsKey = await scopedStorageKey(page, 'potionui_tabs_state');
	await expect
		.poll(() =>
			page.evaluate(({ id, key }) => {
				const raw = localStorage.getItem(key);
				const state = raw ? JSON.parse(raw) : null;
				return Boolean(state?.tabs?.some((t: { selectedPreset: unknown }) => JSON.stringify(t.selectedPreset ?? '').includes(id)));
			}, { id: presetId, key: tabsKey })
		)
		.toBe(true);
}

export async function selectPresetInPicker(page: Page, preset: DiscoveredPreset): Promise<boolean> {
	const pickerTrigger = page.locator('button[aria-haspopup="dialog"]').first();
	await pickerTrigger.click();
	const presetList = page.getByRole('listbox', { name: 'Presets' });
	const opened = await presetList.isVisible({ timeout: 15000 }).catch(() => false);
	if (!opened) return false;
	const entry = presetOption(page, preset.name);
	if ((await entry.count()) === 0) return false;
	await entry.click();
	await expect(page.getByRole('dialog', { name: 'Choose preset' })).toBeHidden();
	await page.waitForLoadState('networkidle');
	const variantSegment = page.getByTestId('preset-variant-switch').getByRole('button').first();
	if ((await variantSegment.count()) > 0) {
		await variantSegment.click();
		const videoOption = page.getByRole('option', { name: /^Video\b/ });
		if ((await videoOption.count()) > 0) await videoOption.first().click();
		else await page.keyboard.press('Escape');
	}
	return true;
}

export async function chooseLayout(page: Page, layout: 'Two panes' | 'Three panes'): Promise<void> {
	const tool = page.locator('[data-context-tool="layout"]');
	if (await tool.isVisible().catch(() => false)) {
		if ((await tool.getAttribute('aria-label')) !== `Layout: ${layout}`) await tool.click();
		await expect(tool).toHaveAttribute('aria-label', `Layout: ${layout}`);
		return;
	}
	await page.getByTestId('context-tools-more').click();
	const item = page.getByRole('menuitem', { name: /^Layout: / });
	if ((await item.getAttribute('aria-label')) !== `Layout: ${layout}`) await item.click();
	else await page.keyboard.press('Escape');
}
