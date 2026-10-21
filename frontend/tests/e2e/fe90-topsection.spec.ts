import { test, expect, type Page } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot, shotPath, apiGet, apiPost } from './helpers';

// Visual evidence — the "1a Instrument Pro" top-section restyle: tabs
// row now carries the session pill + "…" overflow on its right (Simple/
// Advanced moved into that overflow menu alongside the view-layout picker),
// and the left settings pane opens with the relocated preset card + mode
// segmented row (PresetSessionBar deleted, its 15 controls re-homed).
const JOURNEY = 'fe90-topsection';

test('top section — tabs row cluster + relocated preset header', async ({ page }) => {
	await loginAsOwner(page);
	const token = await ownerToken(page);

	const me = await apiGet(page, '/api/auth/me', token);
	const userId = me.data.id as string;

	const list = await apiGet(page, '/api/presets?include_uninstalled=true', token);
	const presets = (list.data || []) as Array<{
		id: string;
		name: string;
		engine?: string;
		category?: string;
		installed?: boolean;
	}>;
	const imagePreset =
		presets.find((p) => /sdxl/i.test(p.name)) ||
		presets.find((p) => p.engine === 'native' && p.category === 'image');

	if (!imagePreset) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return;
	}

	if (!imagePreset.installed) {
		await apiPost(page, `/api/presets/${imagePreset.id}/install`, token);
	}
	await apiPost(page, `/api/presets/${imagePreset.id}/assign`, token, { user_ids: [userId] });

	await page.goto('/generate');
	await page.getByRole('button', { name: 'Choose a preset' }).click();
	await page.getByText(imagePreset.name, { exact: true }).first().click();
	await page.getByRole('button', { name: /Use this preset|Keep selected/ }).click();

	await expect(page.locator('[role="tablist"]').first()).toBeVisible({ timeout: 20000 });

	// 1. Full page — the decisive comparison against generate_1a.png's top strip.
	await screenshot(page, JOURNEY, '01-generate-full');

	const tabBar = page.locator('.tab-bar-container').first();
	await expect(tabBar).toBeVisible();
	await tabBar.screenshot({ path: shotPath(JOURNEY, '02-tabs-row') });

	// 3. Left-column preset card (mode segmented row directly under it).
	const presetCard = page.getByRole('button', { name: imagePreset.name, exact: false }).first();
	await expect(presetCard).toBeVisible();
	const presetHeaderRegion = presetCard.locator(
		'xpath=ancestor::div[contains(@class, "border-b")][1]'
	);
	if ((await presetHeaderRegion.count()) > 0) {
		await presetHeaderRegion.first().screenshot({ path: shotPath(JOURNEY, '03-preset-header') });
	} else {
		await presetCard.screenshot({ path: shotPath(JOURNEY, '03-preset-header') });
	}

	// 4. Open session panel (SessionPill's dropdown — same SessionControl panel).
	const sessionTrigger = page.getByRole('button', { name: 'Session', exact: true });
	await expect(sessionTrigger).toBeVisible();
	await sessionTrigger.click();
	await screenshot(page, JOURNEY, '04-session-panel-open');
	await page.keyboard.press('Escape');

	const tools = page.getByRole('group', { name: 'Generation tools' });
	await expect(tools).toBeVisible();
	await expect(tools.getByRole('button', { name: /^Layout: / })).toBeVisible();
	const simpleFields = tools.getByRole('button', { name: 'Field visibility: Simple' });
	await expect(simpleFields).toBeVisible();
	await screenshot(page, JOURNEY, '05-context-tools');

	await simpleFields.click();
	const advancedFields = tools.getByRole('button', { name: 'Field visibility: Advanced' });
	await expect(advancedFields).toHaveAttribute('aria-pressed', 'true');
	await screenshot(page, JOURNEY, '06-advanced-audience');
	await advancedFields.click();

	// 7. Mobile: the Studio "preset & session" sheet now mounts PresetHeader +
	// SessionPill in place of the deleted PresetSessionBar / mobile carousel
	// panel - not redesigned, just re-homed, so confirm it still renders both.
	await page.setViewportSize({ width: 390, height: 844 });
	const presetSheetTrigger = page.getByRole('button', { name: 'Open preset and session' });
	await expect(presetSheetTrigger).toBeVisible();
	await presetSheetTrigger.click();
	const presetSheet = page.getByRole('dialog', { name: 'Preset and session' });
	await expect(presetSheet).toBeVisible();
	await expect(presetSheet.getByRole('button', { name: imagePreset.name, exact: false }).first()).toBeVisible();
	await screenshot(page, JOURNEY, '07-mobile-preset-panel');
});
