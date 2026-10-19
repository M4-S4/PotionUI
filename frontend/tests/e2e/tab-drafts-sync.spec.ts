import { test, expect, type Page, type Route } from '@playwright/test';
import { loginAsOwner, scopedStorageKey, screenshot } from './helpers';
import { installAndSelectImagePreset } from './presetPreamble';

const JOURNEY = 'tab-drafts-sync';
const OUTBOX_KEY = 'potionui-tab-draft-outbox';
const TABS_KEY = 'potionui_tabs_state';

interface DraftOp {
	op: 'set' | 'remove';
	name: string;
	type?: string;
	value?: string;
}

interface DraftRow {
	variables: Record<string, unknown>;
	rev: number;
}

interface DraftServer {
	rows: Map<string, DraftRow>;
	posts: Array<{ tabId: string; ops: DraftOp[] }>;
	deletes: string[];
	reads: string[];
	mode: { offline: boolean; limit: boolean };
}

async function mockTabDrafts(page: Page): Promise<DraftServer> {
	const server: DraftServer = {
		rows: new Map(),
		posts: [],
		deletes: [],
		reads: [],
		mode: { offline: false, limit: false }
	};
	const ok = (route: Route, data: unknown) =>
		route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ success: true, data }) });
	const fail = (route: Route, status: number, detail: Record<string, unknown>) =>
		route.fulfill({
			status,
			contentType: 'application/json',
			body: JSON.stringify({ detail: { success: false, ...detail } })
		});
	const draftBody = (tabId: string, row: DraftRow) => ({
		tab_id: tabId,
		doc: { variables: row.variables },
		rev: row.rev,
		created_at: null,
		updated_at: null
	});

	await page.route(/\/api\/tab-drafts(\/[^?]*)?(\?.*)?$/, async (route) => {
		if (server.mode.offline) return route.abort('failed');
		const request = route.request();
		const path = new URL(request.url()).pathname.replace(/^.*\/api\/tab-drafts\/?/, '');
		const [tabId, action] = path.split('/');
		const method = request.method();

		if (!tabId && method === 'GET') {
			server.reads.push('list');
			return ok(
				route,
				[...server.rows.entries()].map(([id, row]) => ({
					tab_id: id,
					variable_count: Object.keys(row.variables).length,
					rev: row.rev,
					updated_at: null
				}))
			);
		}
		if (tabId && !action && method === 'GET') {
			server.reads.push(tabId);
			const row = server.rows.get(tabId);
			return row ? ok(route, draftBody(tabId, row)) : fail(route, 404, { error: 'tab_draft_not_found' });
		}
		if (tabId && !action && method === 'DELETE') {
			server.deletes.push(tabId);
			return server.rows.delete(tabId)
				? ok(route, { deleted: true })
				: fail(route, 404, { error: 'tab_draft_not_found' });
		}
		if (tabId && action === 'ops' && method === 'POST') {
			if (server.mode.limit) {
				return fail(route, 409, { error: 'tab_draft_limit', message: 'Too many drafts', limit: 200 });
			}
			const ops = (request.postDataJSON().ops ?? []) as DraftOp[];
			server.posts.push({ tabId, ops });
			const row = server.rows.get(tabId) ?? { variables: {}, rev: 0 };
			for (const op of ops) {
				if (op.op === 'remove') delete row.variables[op.name];
				else row.variables[op.name] = { type: 'text', value: op.value ?? '' };
			}
			row.rev += 1;
			server.rows.set(tabId, row);
			return ok(route, draftBody(tabId, row));
		}
		return fail(route, 405, { error: 'unmocked' });
	});

	return server;
}

async function openVariables(page: Page) {
	const variablesButton = page.getByRole('button', { name: 'Variables' });
	await variablesButton.waitFor({ state: 'visible', timeout: 15000 });
	await variablesButton.click();
	const modal = page.getByRole('dialog').filter({ hasText: 'Prompt variables' });
	await expect(modal).toBeVisible({ timeout: 10000 });
	return modal;
}

async function selectPresetOrSkip(page: Page): Promise<boolean> {
	const preset = await installAndSelectImagePreset(page);
	if (!preset) {
		test.skip(true, 'No native image preset available on this throwaway instance.');
		return false;
	}
	return true;
}

test('edits sync, reload rehydrates, offline outbox flushes, closing a tab deletes the draft', async ({ page }) => {
	test.setTimeout(180000);
	const server = await mockTabDrafts(page);
	await loginAsOwner(page);
	if (!(await selectPresetOrSkip(page))) return;

	let modal = await openVariables(page);
	await modal.getByLabel('Variable name').fill('style');
	await modal.getByLabel('Variable value').fill('oil painting');
	await modal.getByRole('button', { name: 'Done' }).click();

	await expect.poll(() => server.posts.length, { timeout: 10000 }).toBeGreaterThan(0);
	await expect
		.poll(() => [...server.rows.values()][0]?.variables, { timeout: 10000 })
		.toEqual({ style: { type: 'text', value: 'oil painting' } });
	const tabId = server.posts[0].tabId;
	expect(server.posts.every((post) => post.tabId === tabId)).toBe(true);

	const tabsKey = await scopedStorageKey(page, TABS_KEY);
	await expect
		.poll(async () =>
			page.evaluate((key) => {
				const raw = localStorage.getItem(key);
				return raw ? JSON.parse(raw).tabs.some((tab: Record<string, unknown>) => 'variables' in tab) : null;
			}, tabsKey)
		)
		.toBe(false);

	server.mode.offline = true;
	modal = await openVariables(page);
	await modal.getByLabel('Variable value').first().fill('watercolor');
	await modal.getByRole('button', { name: 'Done' }).click();

	const outboxKey = await scopedStorageKey(page, OUTBOX_KEY);
	await expect
		.poll(
			async () =>
				page.evaluate((key) => {
					const raw = localStorage.getItem(key);
					return raw ? Object.keys(JSON.parse(raw).ops ?? {}) : [];
				}, outboxKey),
			{ timeout: 10000 }
		)
		.toEqual([tabId]);
	expect(server.rows.get(tabId)?.variables).toEqual({ style: { type: 'text', value: 'oil painting' } });

	server.mode.offline = false;
	await page.evaluate(() => window.dispatchEvent(new Event('online')));
	await expect
		.poll(() => server.rows.get(tabId)?.variables, { timeout: 15000 })
		.toEqual({ style: { type: 'text', value: 'watercolor' } });
	await expect
		.poll(async () =>
			page.evaluate((key) => {
				const raw = localStorage.getItem(key);
				return raw ? Object.keys(JSON.parse(raw).ops ?? {}).length : 0;
			}, outboxKey)
		)
		.toBe(0);

	server.reads.length = 0;
	await page.reload();
	await expect.poll(() => server.reads, { timeout: 15000 }).toEqual(['list', tabId]);
	modal = await openVariables(page);
	await expect(modal.getByLabel('Variable value').first()).toHaveValue('watercolor');
	await modal.getByRole('button', { name: 'Done' }).click();

	await page.getByRole('button', { name: 'Add new tab' }).click();
	await page.locator('button.book-tab', { hasText: 'Generation 1' }).locator('.close-button').click({ force: true });
	const confirm = page.getByRole('button', { name: 'Confirm', exact: true });
	if (await confirm.isVisible().catch(() => false)) await confirm.click();
	await expect.poll(() => server.deletes, { timeout: 10000 }).toEqual([tabId]);
	expect(server.rows.has(tabId)).toBe(false);
});

test('a draft limit shows a warning toast and keeps the edit queued', async ({ page }) => {
	test.setTimeout(120000);
	const server = await mockTabDrafts(page);
	server.mode.limit = true;
	await loginAsOwner(page);
	if (!(await selectPresetOrSkip(page))) return;

	const modal = await openVariables(page);
	await modal.getByLabel('Variable name').fill('style');
	await modal.getByLabel('Variable value').fill('oil painting');
	await modal.getByRole('button', { name: 'Done' }).click();

	const toast = page.getByText(/Close old tabs/);
	await expect(toast).toBeVisible({ timeout: 15000 });
	expect(server.posts).toHaveLength(0);

	const outboxKey = await scopedStorageKey(page, OUTBOX_KEY);
	const queued = await page.evaluate((key) => {
		const raw = localStorage.getItem(key);
		return raw ? Object.values(JSON.parse(raw).ops ?? {}).length : 0;
	}, outboxKey);
	expect(queued).toBe(1);

	await page.setViewportSize({ width: 1440, height: 900 });
	await screenshot(page, JOURNEY, 'limit-toast-1440');
	await page.setViewportSize({ width: 390, height: 844 });
	await expect(toast).toBeVisible();
	await screenshot(page, JOURNEY, 'limit-toast-390');
});
