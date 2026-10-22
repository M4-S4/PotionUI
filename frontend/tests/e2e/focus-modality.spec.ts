import { test, expect, type Locator, type Page, type WebSocketRoute } from '@playwright/test';
import { loginAsOwner } from './helpers';
import { makeModelFieldsOptional } from './generationMocks';
import { discoverAndAssignImagePreset, presetOption } from './presetPreamble';

const GENERATION_ID = '01FOCUSMODALITY00000000001';
const NO_PRESET = 'No native image preset available on this throwaway instance.';
const SHORTCUT = 'Control+Enter';
const PROMPT = 'a glowing potion bottle on a wooden alchemist bench';
const GENERATE = '.generation-panel .generate-button';

async function stubGeneration(page: Page, sockets: WebSocketRoute[]) {
	await page.routeWebSocket(/\/ws\/generation/, (ws) => {
		sockets.push(ws);
		ws.send(JSON.stringify({ type: 'connection_established', client_id: 'e2e-mock' }));
		ws.onMessage((raw) => {
			const msg = JSON.parse(raw as string);
			if (msg.type === 'subscribe_generation') {
				ws.send(JSON.stringify({ type: 'subscribed', generation_id: msg.generation_id }));
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
}

async function signalColor(page: Page): Promise<string> {
	return page.evaluate(() => {
		const probe = document.createElement('div');
		probe.style.color = 'rgb(var(--signal))';
		document.body.appendChild(probe);
		const value = getComputedStyle(probe).color;
		probe.remove();
		return value;
	});
}

async function ring(el: Locator) {
	return el.evaluate((node) => {
		const style = getComputedStyle(node);
		return { outlineStyle: style.outlineStyle, outlineColor: style.outlineColor, boxShadow: style.boxShadow };
	});
}

async function expectNoOutline(el: Locator) {
	await expect.poll(async () => (await ring(el)).outlineStyle, { timeout: 5000 }).toBe('none');
}

async function expectNoRing(el: Locator) {
	await expect
		.poll(
			async () => {
				const state = await ring(el);
				return `${state.outlineStyle}|${state.boxShadow}`;
			},
			{ timeout: 5000 }
		)
		.toBe('none|none');
}

async function expectModality(page: Page, modality: 'pointer' | 'keyboard') {
	await expect(page.locator('html')).toHaveAttribute('data-input', modality);
}

async function tabUntil(page: Page, selector: string) {
	await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
	for (let hops = 0; hops < 200; hops++) {
		await page.keyboard.press('Tab');
		const reached = await page.evaluate((sel) => document.activeElement?.matches(sel) ?? false, selector);
		if (reached) return;
	}
	throw new Error(`Tab never reached ${selector}`);
}

async function openDesktop(page: Page, typePrompt: boolean, sockets: WebSocketRoute[] = []): Promise<boolean> {
	await stubGeneration(page, sockets);
	await loginAsOwner(page);
	const preset = await discoverAndAssignImagePreset(page);
	if (!preset) return false;
	await page.goto('/generate');
	await page.getByRole('button', { name: 'Choose a preset' }).click();
	await presetOption(page, preset.name).click();
	await expect(page.getByRole('dialog', { name: 'Choose preset' })).toBeHidden();
	if (typePrompt) {
		const editor = page.locator('[contenteditable="true"]').first();
		await expect(editor).toBeVisible({ timeout: 20000 });
		await editor.click();
		await page.keyboard.type(PROMPT);
	}
	await expect(page.locator(GENERATE)).toBeVisible({ timeout: 20000 });
	return true;
}

async function openPhone(page: Page): Promise<boolean> {
	await stubGeneration(page, []);
	await loginAsOwner(page);
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
	await page.getByRole('button', { name: 'Prompt', exact: true }).click();
	const editor = page.locator('[contenteditable="true"]').first();
	await expect(editor).toBeVisible({ timeout: 8000 });
	await editor.click();
	await page.keyboard.type(PROMPT);
	await page.keyboard.press('Escape');
	await expect(page.getByRole('dialog', { name: 'Prompt', exact: true })).toHaveCount(0);
	await expect(page.locator(GENERATE)).toBeVisible({ timeout: 15000 });
	return true;
}

async function checkGenerateRing(page: Page) {
	const generate = page.locator(GENERATE);
	await expect(generate).toBeEnabled({ timeout: 15000 });
	const signal = await signalColor(page);

	await page.keyboard.press(SHORTCUT);
	await generate.click();
	await expectModality(page, 'pointer');
	await expectNoRing(generate);
	await page.keyboard.press(SHORTCUT);
	await expectNoRing(generate);

	await tabUntil(page, GENERATE);
	await expectModality(page, 'keyboard');
	await expect.poll(async () => (await ring(generate)).outlineStyle, { timeout: 5000 }).toBe('solid');
	expect((await ring(generate)).outlineColor).toBe(signal);

	await generate.click();
	await expectModality(page, 'pointer');
	await expectNoRing(generate);
}

test.describe('desktop', () => {
	test.use({ viewport: { width: 1440, height: 900 } });

	test('Generate shows no ring after a click or a shortcut key, and a signal ring when reached with Tab', async ({ page }) => {
		if (!(await openDesktop(page, true))) {
			test.skip(true, NO_PRESET);
			return;
		}
		await checkGenerateRing(page);
	});

	test('the floating workbench draws no outline after w and a click inside it', async ({ page }) => {
		if (!(await openDesktop(page, false))) {
			test.skip(true, NO_PRESET);
			return;
		}
		await page.keyboard.press('w');
		const dialog = page.getByRole('dialog', { name: 'Workbench' });
		await expect(dialog).toBeVisible({ timeout: 10000 });

		await dialog.click({ position: { x: 6, y: 6 } });
		await expect(dialog).toBeFocused();
		await expectModality(page, 'pointer');
		await page.keyboard.press(SHORTCUT);
		await expectNoOutline(dialog);

		const handle = dialog.getByRole('button', { name: 'Resize workbench width' });
		const box = await handle.boundingBox();
		expect(box).not.toBeNull();
		const x = box!.x + box!.width / 2;
		const y = box!.y + box!.height / 2;
		await page.mouse.move(x, y);
		await page.mouse.down();
		await page.mouse.move(x + 12, y);
		await page.mouse.up();
		await page.keyboard.press(SHORTCUT);
		await expectNoOutline(handle);
	});

	test('a framed field keeps its ring when the input is clicked and shows none on its button after a click', async ({ page }) => {
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await loginAsOwner(page);
		await page.goto('/admin?tab=models');
		await expect(page.getByPlaceholder(/Search by filename/).first()).toBeVisible({ timeout: 15000 });
		await page.evaluate(() => {
			const host = document.createElement('div');
			host.style.cssText = 'position:fixed;top:80px;left:40px;width:380px;z-index:9999;padding:16px;background:rgb(var(--surface-1))';
			host.innerHTML =
				'<div id="fx-frame" class="field-frame input" style="display:flex;gap:8px">' +
				'<input id="fx-num" type="number" value="5" /><button id="fx-dice" type="button">+</button></div>';
			document.body.appendChild(host);
		});
		const frame = page.locator('#fx-frame');
		const dice = page.locator('#fx-dice');

		await page.keyboard.press(SHORTCUT);
		await page.locator('#fx-num').click();
		await expectModality(page, 'pointer');
		await expect.poll(async () => (await ring(frame)).boxShadow, { timeout: 5000 }).not.toBe('none');

		await dice.click();
		await page.keyboard.press(SHORTCUT);
		await expectNoRing(frame);
		await expectNoOutline(dice);
	});

	test('the deck More menu opened from the keyboard lands on its first visible row', async ({ page }) => {
		const sockets: WebSocketRoute[] = [];
		if (!(await openDesktop(page, true, sockets))) {
			test.skip(true, NO_PRESET);
			return;
		}
		const generate = page.locator(GENERATE);
		await expect(generate).toBeEnabled({ timeout: 15000 });
		await generate.click();
		await expect.poll(() => sockets.length, { timeout: 15000 }).toBeGreaterThan(0);
		sockets[sockets.length - 1].send(
			JSON.stringify({
				type: 'generation_complete',
				data: {
					id: GENERATION_ID,
					generation_id: GENERATION_ID,
					status: 'completed',
					progress: 1.0,
					created_at: new Date().toISOString(),
					completed_at: new Date().toISOString()
				}
			})
		);
		await expect(generate).toHaveAttribute('aria-label', 'Generate', { timeout: 10000 });

		await page.getByRole('button', { name: 'More generation actions' }).focus();
		await page.keyboard.press('Enter');
		const details = page.getByRole('menuitem', { name: /Run details/ });
		await expect(details).toBeVisible({ timeout: 5000 });
		await expect(details).toBeFocused();
		await expect(page.getByRole('menuitem', { name: /^Queue/ })).toHaveCount(0);

		await page.keyboard.press('ArrowDown');
		await expect(details).toBeFocused();
		await expect
			.poll(() => page.evaluate(() => (document.activeElement?.getClientRects().length ?? 0) > 0), { timeout: 5000 })
			.toBe(true);
	});
});

test.describe('phone', () => {
	test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });

	test('Generate shows no ring after a tap or a shortcut key, and a signal ring when reached with Tab', async ({ page }) => {
		if (!(await openPhone(page))) {
			test.skip(true, NO_PRESET);
			return;
		}
		await checkGenerateRing(page);
	});

	test('a menu opened with a tap highlights its row once an arrow key moves focus', async ({ page }) => {
		if (!(await openPhone(page))) {
			test.skip(true, NO_PRESET);
			return;
		}
		await page.getByRole('button', { name: 'More generation actions' }).click();
		const items = page.locator('[role="menuitem"]:not(:disabled)');
		await expect(items.first()).toBeVisible({ timeout: 5000 });
		await expect(items.first()).toBeFocused();
		await expectModality(page, 'pointer');
		const fill = (el: Locator) => el.evaluate((node) => getComputedStyle(node).backgroundColor);
		const resting = await fill(items.first());

		await page.keyboard.press('ArrowDown');
		await expectModality(page, 'keyboard');
		const focused = page.locator('[role="menuitem"]:focus');
		await expect(focused).toHaveCount(1);
		await expect.poll(() => fill(focused), { timeout: 5000 }).not.toBe(resting);
	});
});
