import { test, expect, type Page } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';
import { discoverAndAssignImagePreset, presetOption } from './presetPreamble';

const JOURNEY = 'mobile-generate';

// Phone-viewport shell contract for the generate page's Studio shell (the
// camera-app posture: full-bleed canvas, a floating top bar, a floating dock
// with the prompt pill + shutter, and every heavy control — prompt, settings,
// preset/session, chat — living in a bottom sheet over the canvas). Guards
// the class of regression where the page overflows behind the fixed bottom
// tab bar (100vh shells inside a pb-reserved main), the dock ends up
// underneath the tab bar instead of anchored above it, or a sheet's nested
// overlay (the preset picker) resolves its `position: fixed` against the
// wrong containing block instead of the viewport.
test.use({ viewport: { width: 375, height: 812 }, hasTouch: true });

test('generate page fits and shows the Studio shell at phone width', async ({ page }) => {
	await loginAsOwner(page);
	await page.goto('/generate');

	// Mobile chrome is up: full-bleed canvas + floating top bar + dock, and the
	// bottom tab bar, no desktop sidebar.
	const canvas = page.locator('.studio-canvas');
	await expect(canvas).toBeVisible({ timeout: 15000 });
	await expect(page.locator('.studio-topbar')).toBeVisible();
	await expect(page.locator('.studio-dock')).toBeVisible();
	const tabBar = page.locator('nav.fixed.bottom-0');
	await expect(tabBar.filter({ hasText: 'Generate' })).toBeVisible();

	// No page-level scrolling in either axis: horizontal overflow was how every
	// layout bug degraded (main is a scroller), and vertical overflow means the
	// shell slid behind the fixed tab bar. scrollbar-gutter can make scrollWidth
	// come up SHORT of clientWidth, so assert "no overflow", not equality.
	const { overflowX, overflowY } = await page.evaluate(() => {
		const el = document.scrollingElement!;
		return {
			overflowX: el.scrollWidth - el.clientWidth,
			overflowY: el.scrollHeight - el.clientHeight
		};
	});
	expect(overflowX, 'page must not scroll horizontally').toBeLessThanOrEqual(0);
	expect(overflowY, 'page shell must fit the viewport (no scroll behind the tab bar)').toBeLessThanOrEqual(0);

	// The canvas is full-bleed: it fills the content area left-to-right, flush
	// with the viewport's left edge (the regression this replaces was panels
	// sized in vw instead of container %, drifting out of alignment).
	const canvasBox = await canvas.boundingBox();
	expect(canvasBox, 'canvas has a box').toBeTruthy();
	expect(Math.abs(canvasBox!.x), 'canvas flush with viewport left edge').toBeLessThanOrEqual(1);
	expect(Math.abs(canvasBox!.width - 375), 'canvas spans the full viewport width').toBeLessThanOrEqual(1);

	// Top bar: tab pill + preset/mode pill + AI chat trigger.
	await expect(page.getByRole('button', { name: 'Open tabs' })).toBeVisible();
	await expect(page.getByRole('button', { name: 'Open preset and session' })).toBeVisible();
	await expect(page.getByRole('button', { name: 'Open AI chat' })).toBeVisible();

	// Dock: prompt pill + shutter, anchored ABOVE the fixed bottom tab bar, not
	// underneath or overlapping it.
	const promptPill = page.getByRole('button', { name: 'Prompt' });
	const shutter = page.locator('.generation-panel .generate-button');
	await expect(promptPill).toBeVisible();
	await expect(shutter).toBeVisible();
	const shutterBox = await shutter.boundingBox();
	const tabBarBox = await tabBar.boundingBox();
	expect(shutterBox!.y + shutterBox!.height, 'shutter sits above the tab bar').toBeLessThanOrEqual(tabBarBox!.y + 1);

	// The bottom tab bar must sit fully inside the viewport.
	expect(tabBarBox!.y + tabBarBox!.height, 'tab bar bottom edge on-screen').toBeLessThanOrEqual(812 + 1);
	expect(tabBarBox!.y, 'tab bar visible above viewport bottom').toBeGreaterThan(700);

	await screenshot(page, JOURNEY, 'studio-idle');
});

test('prompt pill opens the prompt sheet; Escape closes it', async ({ page }) => {
	await loginAsOwner(page);
	await page.goto('/generate');

	await expect(page.locator('.studio-dock')).toBeVisible({ timeout: 15000 });
	await page.getByRole('button', { name: 'Prompt' }).click();

	const promptSheet = page.getByRole('dialog', { name: 'Prompt' });
	await expect(promptSheet).toBeVisible({ timeout: 5000 });
	// The default (segmented) prompt editor — reachable with no preset chosen,
	// since PromptSection falls back to it whenever no other mode (relay,
	// director, multi-prompt) is active.
	await expect(promptSheet.getByRole('list', { name: 'Positive segments' })).toBeVisible();
	await screenshot(page, JOURNEY, 'prompt-sheet');

	await page.keyboard.press('Escape');
	// The sheet is destroyed on close (not just hidden) — its document-level
	// Escape listener must go with it, or a second Escape press elsewhere in
	// the app would hit a detached instance.
	await expect(promptSheet).toHaveCount(0);
});

test('Settings pill opens the settings sheet; backdrop click closes it', async ({ page }) => {
	await loginAsOwner(page);
	await page.goto('/generate');

	await expect(page.locator('.studio-dock')).toBeVisible({ timeout: 15000 });
	await page.getByRole('button', { name: 'Settings', exact: true }).click();

	const settingsSheet = page.getByRole('dialog', { name: 'Settings' });
	await expect(settingsSheet).toBeVisible({ timeout: 5000 });
	await screenshot(page, JOURNEY, 'settings-sheet');

	// Click the backdrop, not the sheet panel — the sheet is anchored to the
	// bottom (max-height 88%), leaving the top of the viewport as backdrop.
	await page.mouse.click(20, 20);
	await expect(settingsSheet).toHaveCount(0);
});

test('overlay opened from inside a Studio sheet covers the full viewport', async ({ page }) => {
	await loginAsOwner(page);
	await page.goto('/generate');

	await expect(page.locator('.studio-dock')).toBeVisible({ timeout: 15000 });
	await page.getByRole('button', { name: 'Open preset and session' }).click();
	const presetSheet = page.getByRole('dialog', { name: 'Preset and session' });
	await expect(presetSheet).toBeVisible({ timeout: 5000 });

	await presetSheet.locator('button[aria-haspopup="dialog"]').first().click();

	const picker = page.getByRole('dialog', { name: 'Choose preset' });
	await expect(picker).toBeVisible();
	const viewport = page.viewportSize()!;
	const box = await picker.boundingBox();
	expect(box, 'preset picker has a box').toBeTruthy();
	expect(Math.abs(box!.x), 'picker left edge at viewport origin').toBeLessThanOrEqual(20);
	expect(Math.abs(box!.y), 'picker top edge at viewport origin').toBeLessThanOrEqual(1);
	expect(Math.abs(box!.width - viewport.width), 'picker spans the full viewport width').toBeLessThanOrEqual(20);
	expect(Math.abs(box!.height - viewport.height), 'picker spans the full viewport height').toBeLessThanOrEqual(1);
});

test('a dropped mode-list request does not strand the settings sheet forever', async ({ page }) => {
	// Regression: picking a preset kicks off GET /api/presets/{id}/modes to
	// auto-select a mode. That fetch is only ever retried by the fallout of
	// some OTHER reactive update (tab switch, generation event, ...) - nothing
	// re-triggers it on its own. On a flaky mobile connection a single dropped
	// request left the tab with selectedMode permanently null: the settings
	// sheet stuck with no form fields (GenerationFormPane never mounts without
	// a selected mode) and Panel 0's session pill stuck disabled ("Session
	// unavailable") forever, with no user-visible way to recover short of
	// reselecting the preset from scratch.
	await loginAsOwner(page);

	let modesCallCount = 0;
	await page.route('**/api/presets/*/modes', async (route) => {
		modesCallCount++;
		if (modesCallCount === 1) {
			await route.abort('failed');
			return;
		}
		await route.continue();
	});

	// installAndSelectImagePreset assumes the "Choose a preset" trigger is
	// immediately reachable, which is only true on desktop - on mobile it
	// lives inside the preset & session sheet, closed by default. Install/
	// assign via its API calls, then drive the picker through the mobile
	// chrome ourselves, same as the overlay test above.
	const preset = await discoverAndAssignImagePreset(page);
	expect(preset, 'a native image preset must exist to select').toBeTruthy();

	await page.goto('/generate');
	await expect(page.locator('.studio-dock')).toBeVisible({ timeout: 15000 });

	await page.getByRole('button', { name: 'Open preset and session' }).click();
	const presetSheet = page.getByRole('dialog', { name: 'Preset and session' });
	await expect(presetSheet).toBeVisible({ timeout: 5000 });

	const pickerTrigger = presetSheet.locator('button[aria-haspopup="dialog"]', { hasText: 'Choose a preset' });
	await expect(pickerTrigger).toBeVisible({ timeout: 15000 });
	await pickerTrigger.click();
	const presetList = page.locator('[role="listbox"][aria-label="Presets"]');
	await expect(presetList).toBeVisible({ timeout: 15000 });
	await presetOption(page, preset!.name).click();
	await expect(page.getByRole('dialog', { name: 'Choose preset' })).toBeHidden();

	await page.keyboard.press('Escape');
	await expect(presetSheet).toHaveCount(0);

	await page.getByRole('button', { name: 'Settings', exact: true }).click();
	const settingsSheet = page.getByRole('dialog', { name: 'Settings' });
	await expect(settingsSheet).toBeVisible({ timeout: 5000 });

	// A form field actually renders — proof the retried /modes request landed
	// and selectedMode got set, not just that the sheet opened.
	await expect(settingsSheet.locator('input, select, textarea').first()).toBeVisible({ timeout: 10000 });
	expect(modesCallCount, 'the dropped /modes request must have been retried').toBeGreaterThan(1);
});

test('the generation deck sits between the Studio canvas and the tab bar', async ({ page }) => {
	await loginAsOwner(page);
	await page.goto('/generate');

	const deck = page.locator('.generation-panel');
	await expect(deck).toBeVisible({ timeout: 15000 });
	await expect(page.locator('.studio-dock')).toBeVisible();

	const tabBarBox = await page.locator('nav.fixed.bottom-0').boundingBox();
	const deckBox = await deck.boundingBox();
	const canvasBox = await page.locator('.studio-canvas').boundingBox();
	const dockBox = await page.locator('.studio-dock').boundingBox();

	expect(Math.abs(deckBox!.y + deckBox!.height - tabBarBox!.y), 'deck bottom meets the tab bar').toBeLessThanOrEqual(1);
	expect(Math.abs(deckBox!.height - 140), 'deck is 140px tall').toBeLessThanOrEqual(1);
	expect(Math.abs(deckBox!.width - 375), 'deck spans the viewport').toBeLessThanOrEqual(1);
	expect(canvasBox!.y + canvasBox!.height, 'canvas ends above the deck').toBeLessThanOrEqual(deckBox!.y + 1);
	expect(dockBox!.y + dockBox!.height, 'prompt row ends above the deck').toBeLessThanOrEqual(deckBox!.y + 1);

	await expect(page.locator('[role="progressbar"]')).toHaveCount(1);
	await expect(page.locator('.studio-dock svg circle')).toHaveCount(0);

	await screenshot(page, JOURNEY, 'studio-deck-idle');
});

test('X/Y, Generate and Continuous share one row in that order', async ({ page }) => {
	await loginAsOwner(page);
	await page.goto('/generate');

	const controls = page.locator('.generation-panel .run-cluster button');
	await expect(controls).toHaveCount(3, { timeout: 15000 });
	await expect(controls.nth(0)).toHaveAttribute('data-compare-toggle', '');
	await expect(controls.nth(1)).toHaveClass(/generate-button/);

	const boxes = await controls.evaluateAll((elements) =>
		elements.map((element) => {
			const rect = element.getBoundingClientRect();
			return { x: rect.x, centerY: rect.y + rect.height / 2, width: rect.width, height: rect.height };
		})
	);
	expect(boxes[0].x).toBeLessThan(boxes[1].x);
	expect(boxes[1].x).toBeLessThan(boxes[2].x);
	expect(Math.abs(boxes[0].centerY - boxes[1].centerY)).toBeLessThanOrEqual(1);
	expect(Math.abs(boxes[2].centerY - boxes[1].centerY)).toBeLessThanOrEqual(1);
	expect(Math.abs(boxes[1].width - 60)).toBeLessThanOrEqual(1);
	expect(boxes[0].width).toBeGreaterThanOrEqual(44);
	expect(boxes[2].width).toBeGreaterThanOrEqual(44);
});

test('the Settings button sits right after the prompt pill on the same row', async ({ page }) => {
	await loginAsOwner(page);
	await page.goto('/generate');

	const prompt = page.getByRole('button', { name: 'Prompt', exact: true });
	const settings = page.getByRole('button', { name: 'Settings', exact: true });
	await expect(settings).toBeVisible({ timeout: 15000 });

	const promptBox = await prompt.boundingBox();
	const settingsBox = await settings.boundingBox();
	expect(Math.abs(settingsBox!.width - 44)).toBeLessThanOrEqual(1);
	expect(Math.abs(settingsBox!.height - 44)).toBeLessThanOrEqual(1);
	expect(settingsBox!.x).toBeGreaterThanOrEqual(promptBox!.x + promptBox!.width - 1);
	expect(
		Math.abs(settingsBox!.y + settingsBox!.height / 2 - (promptBox!.y + promptBox!.height / 2)),
		'Settings is centred on the prompt pill'
	).toBeLessThanOrEqual(1);
});

test('Studio chrome and the deck draw no text under 12px', async ({ page }) => {
	await loginAsOwner(page);
	await page.goto('/generate');
	await expect(page.locator('.generation-panel')).toBeVisible({ timeout: 15000 });

	const tooSmall = await page.evaluate(() => {
		const roots = ['.studio-topbar', '.studio-dock', '.generation-panel']
			.map((selector) => document.querySelector(selector))
			.filter((root): root is Element => !!root);
		const small: string[] = [];
		for (const root of roots) {
			for (const element of root.querySelectorAll('*')) {
				const ownText = [...element.childNodes].some((node) => node.nodeType === 3 && node.textContent?.trim());
				if (!ownText || element.getClientRects().length === 0) continue;
				const style = getComputedStyle(element);
				if (style.visibility === 'hidden') continue;
				if (parseFloat(style.fontSize) < 12) small.push(`${element.className} ${style.fontSize}`);
			}
		}
		return small;
	});
	expect(tooSmall).toEqual([]);
});

test('a lost WebSocket shows its warning between the Studio dock and the deck with no blank band', async ({ page }) => {
	await page.addInitScript(() => {
		const Native = window.WebSocket;
		const Stalled = function (url: string, protocols?: string | string[]) {
			if (!String(url).includes('/ws/generation')) return new Native(url, protocols);
			return Object.assign(new EventTarget(), { readyState: 0, close() {}, send() {} });
		} as unknown as typeof WebSocket;
		Object.assign(Stalled, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
		window.WebSocket = Stalled;
	});
	await loginAsOwner(page);
	await page.goto('/generate');

	const warning = page.locator('div.border-warning\\/25', { hasText: 'WebSocket disconnected' });
	await expect(warning).toBeVisible({ timeout: 15000 });
	await expect(page.locator('.generation-panel')).toBeVisible();

	const dockBox = await page.locator('.studio-dock').boundingBox();
	const canvasBox = await page.locator('.studio-canvas').boundingBox();
	const warningBox = await warning.boundingBox();
	const deckBox = await page.locator('.generation-panel').boundingBox();

	expect(dockBox!.y + dockBox!.height, 'prompt row ends at the warning').toBeLessThanOrEqual(warningBox!.y + 1);
	expect(canvasBox!.y + canvasBox!.height, 'canvas ends at the warning').toBeLessThanOrEqual(warningBox!.y + 1);
	expect(warningBox!.y + warningBox!.height, 'warning ends at the deck').toBeLessThanOrEqual(deckBox!.y + 1);
	expect(Math.abs(deckBox!.y - (warningBox!.y + warningBox!.height)), 'no blank band between warning and deck').toBeLessThanOrEqual(1);
	await screenshot(page, JOURNEY, 'studio-deck-disconnected');
});

async function offerInstall(page: Page) {
	await page.evaluate(() => {
		const event: any = new Event('beforeinstallprompt', { cancelable: true });
		event.prompt = async () => {};
		event.userChoice = new Promise(() => {});
		window.dispatchEvent(event);
	});
}

test('the install banner sits above the deck on Generate, below its drawers', async ({ page }) => {
	await loginAsOwner(page);
	await page.goto('/generate');
	await expect(page.locator('.generation-panel')).toBeVisible({ timeout: 15000 });
	await offerInstall(page);

	const banner = page.locator('[data-install-banner]');
	await expect(banner).toBeVisible();
	const bannerBox = await banner.boundingBox();
	const deckBox = await page.locator('.generation-panel').boundingBox();
	expect(bannerBox!.y + bannerBox!.height, 'banner ends above the deck').toBeLessThanOrEqual(deckBox!.y + 1);
	expect(Number(await banner.evaluate((element) => getComputedStyle(element).zIndex)), 'banner stacks under the deck').toBeLessThan(30);
	await screenshot(page, JOURNEY, 'install-banner-generate');
});

test('the install banner sits on the tab bar away from Generate', async ({ page }) => {
	await loginAsOwner(page);
	await page.goto('/history');
	await expect(page.locator('nav.fixed.bottom-0')).toBeVisible({ timeout: 15000 });
	await offerInstall(page);

	const banner = page.locator('[data-install-banner]');
	await expect(banner).toBeVisible();
	const bannerBox = await banner.boundingBox();
	const tabBarBox = await page.locator('nav.fixed.bottom-0').boundingBox();
	expect(Math.abs(bannerBox!.y + bannerBox!.height - tabBarBox!.y), 'banner rests on the tab bar').toBeLessThanOrEqual(1);
	expect(Number(await banner.evaluate((element) => getComputedStyle(element).zIndex))).toBe(30);
});
