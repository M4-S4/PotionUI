import { test, expect, type Page, type Locator } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot } from './helpers';
import { expectPresetPersisted, installAndSelectImagePreset } from './presetPreamble';

const JOURNEY = 'sessions-drawer-long-lists';
const SESSION_COUNT = 25;
const VERSION_COUNT = 40;
const LONG_NAME = 'Long history session';

async function seedSessions(page: Page, presetId: string): Promise<void> {
	const headers = { Authorization: `Bearer ${await ownerToken(page)}` };
	const payload = (prompt: string) => ({ txt2img: { prompt, negativePrompt: '', formData: {} } });

	const created = await page.request.post('/api/sessions/save', {
		headers,
		data: { preset_id: presetId, name: LONG_NAME, data: payload('long history v1') }
	});
	expect(created.ok(), `seed long session: ${created.status()}`).toBeTruthy();
	const longId = (await created.json()).data.id as string;

	for (let i = 2; i <= VERSION_COUNT; i++) {
		const updated = await page.request.put(`/api/sessions/${longId}`, {
			headers,
			data: { name: LONG_NAME, data: payload(`long history v${i} neon alley at night`) }
		});
		expect(updated.ok(), `seed version ${i}: ${updated.status()}`).toBeTruthy();
	}

	for (let i = 1; i <= SESSION_COUNT; i++) {
		const name = `Seeded session ${String(i).padStart(2, '0')}`;
		const res = await page.request.post('/api/sessions/save', {
			headers,
			data: { preset_id: presetId, name, data: payload(`seeded prompt ${i}`) }
		});
		expect(res.ok(), `seed ${name}: ${res.status()}`).toBeTruthy();
	}
}

async function expectInsideViewport(page: Page, drawer: Locator): Promise<void> {
	const viewport = page.viewportSize()!;
	await expect
		.poll(async () => {
			const first = await drawer.boundingBox();
			await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
			return JSON.stringify(first) === JSON.stringify(await drawer.boundingBox());
		})
		.toBe(true);
	const box = await drawer.boundingBox();
	expect(box, 'drawer has a box').not.toBeNull();
	expect(box!.y).toBeGreaterThanOrEqual(0);
	expect(box!.x).toBeGreaterThanOrEqual(0);
	expect(box!.y + box!.height).toBeLessThanOrEqual(viewport.height + 0.5);
	expect(box!.x + box!.width).toBeLessThanOrEqual(viewport.width + 0.5);
}

async function scrollMetrics(locator: Locator) {
	return locator.evaluate((el) => ({ scrollHeight: el.scrollHeight, clientHeight: el.clientHeight, scrollTop: el.scrollTop }));
}

async function openDrawer(page: Page): Promise<Locator> {
	await page.getByTestId('session-trigger').first().click();
	const drawer = page.getByRole('dialog', { name: 'Sessions' });
	await expect(drawer).toBeVisible({ timeout: 10000 });
	return drawer;
}

async function openVersions(page: Page, drawer: Locator, name: string): Promise<void> {
	await drawer.getByRole('button', { name: `More actions for ${name}` }).click();
	await page.getByRole('menuitem', { name: 'Version history' }).click();
	await expect(drawer.getByRole('button', { name: 'Back to sessions' })).toBeVisible({ timeout: 10000 });
}

async function restoreVersion(page: Page, drawer: Locator, index: number): Promise<void> {
	await drawer.locator('[data-version-row]').nth(index).click();
	const discard = page.getByRole('alertdialog');
	const asked = await discard.waitFor({ state: 'visible', timeout: 1500 }).then(
		() => true,
		() => false
	);
	if (asked) {
		await discard.getByRole('button', { name: /^Confirm/ }).click();
		await expect(discard).toBeHidden();
	}
}

async function expectUnderSessionSegment(page: Page, drawer: Locator): Promise<void> {
	const segment = await page.getByTestId('context-session').boundingBox();
	const box = await drawer.boundingBox();
	expect(segment, 'the session segment has a box').not.toBeNull();
	expect(Math.abs(box!.x - segment!.x)).toBeLessThan(1);
	expect(Math.abs(box!.width - 440)).toBeLessThan(1);
	expect(Math.abs(box!.y - (segment!.y + segment!.height + 6))).toBeLessThan(1);
}

test('the sessions drawer stays inside the viewport and scrolls a long list and a long history', async ({ page }) => {
	test.setTimeout(240000);
	await loginAsOwner(page);

	const preset = await installAndSelectImagePreset(page);
	if (!preset) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return;
	}
	await seedSessions(page, preset.id);

	await page.setViewportSize({ width: 1440, height: 900 });
	await page.reload();
	await page.waitForLoadState('networkidle');

	let drawer = await openDrawer(page);
	await expect(drawer.locator('[data-session-row]').first()).toBeVisible({ timeout: 10000 });
	await expect
		.poll(() => drawer.locator('[data-session-row]').count(), { timeout: 10000 })
		.toBeGreaterThanOrEqual(SESSION_COUNT + 1);
	await expectInsideViewport(page, drawer);
	await screenshot(page, JOURNEY, '01-desktop-open');

	const list = drawer.getByTestId('session-list-scroll');
	const before = await scrollMetrics(list);
	expect(before.scrollHeight).toBeGreaterThan(before.clientHeight);
	await list.evaluate((el) => (el.scrollTop = el.scrollHeight));
	const after = await scrollMetrics(list);
	expect(after.scrollTop).toBeGreaterThan(0);
	await expectInsideViewport(page, drawer);
	await expect(drawer.getByLabel('Search sessions')).toBeInViewport();
	await expectUnderSessionSegment(page, drawer);
	await screenshot(page, JOURNEY, '02-desktop-list-scrolled');

	const search = drawer.getByLabel('Search sessions');
	await search.fill('seeded session 07');
	await expect(drawer.locator('[data-session-row]')).toHaveCount(1);
	await search.fill('');

	await drawer.locator('[data-session-row]', { hasText: LONG_NAME }).first().click();
	await expect(drawer).toBeHidden({ timeout: 10000 });
	await expect(page.getByTestId('session-trigger').first()).toContainText(LONG_NAME, { timeout: 10000 });

	drawer = await openDrawer(page);
	await expect(drawer.locator('[data-session-row][aria-current="true"]')).toContainText(LONG_NAME, { timeout: 10000 });
	await expect(drawer.locator('[data-version-row]')).toHaveCount(0);
	await openVersions(page, drawer, LONG_NAME);
	await expect(drawer.locator('[data-version-row]')).toHaveCount(20, { timeout: 10000 });
	const versions = drawer.getByTestId('session-versions-scroll');
	const versionMetrics = await scrollMetrics(versions);
	expect(versionMetrics.scrollHeight).toBeGreaterThan(versionMetrics.clientHeight);
	await expectInsideViewport(page, drawer);
	await screenshot(page, JOURNEY, '03-desktop-versions');

	await drawer.getByRole('button', { name: /Show 20 older/ }).click();
	await expect(drawer.locator('[data-version-row]')).toHaveCount(VERSION_COUNT);
	await expectInsideViewport(page, drawer);
	await screenshot(page, JOURNEY, '04-desktop-all-versions');

	await page.keyboard.press('Escape');
	await expect(drawer.locator('[data-version-row]')).toHaveCount(0);
	await expect(drawer.locator('[data-session-row]').first()).toBeVisible();
	await page.keyboard.press('Escape');
	await expect(drawer).toBeHidden();
	await expect(page.getByTestId('session-trigger').first()).toBeFocused();

	await page.setViewportSize({ width: 390, height: 844 });
	await page.getByRole('button', { name: 'Open preset and session' }).click();
	const sheet = page.getByRole('dialog', { name: 'Preset and session' });
	await expect(sheet).toBeVisible();
	await sheet.locator('button[aria-label="Session"]').click();

	const mobile = page.getByRole('dialog', { name: 'Sessions' });
	await expect(mobile).toBeVisible({ timeout: 10000 });
	const box = await mobile.boundingBox();
	expect(box!.width).toBeGreaterThanOrEqual(389);
	expect(box!.height).toBeGreaterThanOrEqual(843);
	await expectInsideViewport(page, mobile);
	await screenshot(page, JOURNEY, '05-mobile-open');

	const mobileList = mobile.getByTestId('session-list-scroll');
	const mobileBefore = await scrollMetrics(mobileList);
	expect(mobileBefore.scrollHeight).toBeGreaterThan(mobileBefore.clientHeight);
	await mobileList.evaluate((el) => (el.scrollTop = el.scrollHeight));
	expect((await scrollMetrics(mobileList)).scrollTop).toBeGreaterThan(0);
	await expectInsideViewport(page, mobile);
	await screenshot(page, JOURNEY, '06-mobile-list-scrolled');

	await mobileList.evaluate((el) => (el.scrollTop = 0));
	await openVersions(page, mobile, LONG_NAME);
	await expect(mobile.locator('[data-version-row]')).toHaveCount(20, { timeout: 10000 });
	await expectInsideViewport(page, mobile);
	await screenshot(page, JOURNEY, '07-mobile-all-versions');
});

test('the sessions dropdown spans the form column and steps through versions without closing', async ({ page }) => {
	test.setTimeout(240000);
	await loginAsOwner(page);

	const preset = await installAndSelectImagePreset(page);
	if (!preset) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return;
	}
	await seedSessions(page, preset.id);

	for (const { width, height } of [
		{ width: 1280, height: 800 },
		{ width: 1440, height: 900 },
		{ width: 1920, height: 1080 }
	]) {
		await page.setViewportSize({ width, height });
		await page.reload();
		await page.waitForLoadState('networkidle');

		const trigger = page.getByTestId('session-trigger').first();
		await trigger.click();
		let drawer = page.getByRole('dialog', { name: 'Sessions' });
		await expect(drawer).toBeVisible({ timeout: 10000 });
		if (!(await trigger.textContent())?.includes(LONG_NAME)) {
			await drawer.locator('[data-session-row]', { hasText: LONG_NAME }).first().click();
			await expect(drawer).toBeHidden({ timeout: 10000 });
			await expect(trigger).toContainText(LONG_NAME, { timeout: 10000 });
			await trigger.click();
			drawer = page.getByRole('dialog', { name: 'Sessions' });
			await expect(drawer).toBeVisible({ timeout: 10000 });
		}
		await expectUnderSessionSegment(page, drawer);
		await expectInsideViewport(page, drawer);

		const coveredBy = await trigger.evaluate((el) => {
			const rect = el.getBoundingClientRect();
			const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
			return hit === el || el.contains(hit);
		});
		expect(coveredBy, 'the session row above the dropdown stays reachable').toBe(true);
		const tabHit = await page.getByText('Generation 1', { exact: true }).first().evaluate((el) => {
			const rect = el.getBoundingClientRect();
			const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
			return hit === el || el.contains(hit);
		});
		expect(tabHit, 'the tab bar stays reachable').toBe(true);

		await openVersions(page, drawer, LONG_NAME);
		await expect(drawer.locator('[data-version-row]')).toHaveCount(20, { timeout: 10000 });
		await restoreVersion(page, drawer, 1);
		await expect(drawer.locator('[data-version-row]')).toHaveCount(20);
		await restoreVersion(page, drawer, 2);
		await expect(drawer).toBeVisible();
		await expect(drawer.locator('[data-version-row]')).toHaveCount(20);
		await screenshot(page, JOURNEY, `08-versions-stay-open-${width}`);

		await page.keyboard.press('Escape');
		await expect(drawer.locator('[data-version-row]')).toHaveCount(0);
		await page.mouse.click(width - 60, Math.round(height / 3));
		await expect(drawer).toBeHidden({ timeout: 10000 });
	}
});

test('a pinned session moves into the Pinned section and stays there after the drawer is reopened', async ({ page }) => {
	test.setTimeout(240000);
	await loginAsOwner(page);

	const preset = await installAndSelectImagePreset(page);
	if (!preset) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return;
	}
	const headers = { Authorization: `Bearer ${await ownerToken(page)}` };
	const payload = (prompt: string) => ({ txt2img: { prompt, negativePrompt: '', formData: {} } });
	for (const name of ['Pin target', 'Pin bystander', 'Pin current']) {
		const res = await page.request.post('/api/sessions/save', {
			headers,
			data: { preset_id: preset.id, name, data: payload(name) }
		});
		expect(res.ok(), `seed ${name}: ${res.status()}`).toBeTruthy();
	}

	await page.setViewportSize({ width: 1440, height: 900 });
	await expectPresetPersisted(page, preset.id);
	await page.reload();
	await page.waitForLoadState('networkidle');

	let drawer = await openDrawer(page);
	await expect(drawer.locator('[data-session-row]').first()).toBeVisible({ timeout: 10000 });
	await expect(drawer.getByTestId('pinned-head')).toHaveCount(0);

	const pinResponse = page.waitForResponse(
		(response) => response.url().includes('/pin') && response.request().method() === 'PUT'
	);
	await drawer.getByRole('button', { name: 'More actions for Pin target' }).click();
	await page.getByRole('menuitem', { name: 'Pin', exact: true }).click();
	expect((await pinResponse).ok()).toBeTruthy();
	await expect(drawer.getByTestId('pinned-head')).toBeVisible();
	await expect(drawer.getByTestId('pinned-list').locator('[data-session-row]')).toHaveCount(1);
	await expect(drawer.getByTestId('pinned-list')).toContainText('Pin target');
	await screenshot(page, JOURNEY, '06-pinned-section');

	await page.keyboard.press('Escape');
	await expect(drawer).toBeHidden();
	await expectPresetPersisted(page, preset.id);
	await page.reload();
	await page.waitForLoadState('networkidle');

	drawer = await openDrawer(page);
	const pinned = drawer.getByTestId('pinned-list').locator('[data-session-row]');
	await expect(pinned).toHaveCount(1, { timeout: 10000 });
	await expect(pinned.first()).toContainText('Pin target');

	await drawer.getByLabel('Search sessions').fill('bystander');
	await expect(drawer.getByTestId('pinned-head')).toHaveCount(0);
	await drawer.getByLabel('Search sessions').fill('');

	await drawer.getByRole('button', { name: 'More actions for Pin target' }).click();
	await page.getByRole('menuitem', { name: 'Unpin', exact: true }).click();
	await expect(drawer.getByTestId('pinned-head')).toHaveCount(0);
});
