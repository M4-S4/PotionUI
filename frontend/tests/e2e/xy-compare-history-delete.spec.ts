import { test, expect, type Page } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';

const JOURNEY = 'xy-compare-history-delete';
const GRID_ID = '01XYDELETE0000000000000G';
const PNG_1X1 = Buffer.from(
	'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
	'base64'
);

function item(id: string, minutesAgo: number, over: Record<string, unknown> = {}) {
	const created = new Date(Date.now() - minutesAgo * 60_000).toISOString();
	return {
		id,
		preset_id: 'e2e/xy-delete',
		preset_name: 'Compare Delete',
		mode: 'txt2img',
		form_data: { prompt: id },
		status: 'completed',
		progress: 1,
		created_at: created,
		started_at: created,
		completed_at: created,
		updated_at: created,
		rating: 0,
		is_favorite: false,
		tags: [],
		files: [
			{
				id: Number(id.slice(-1)) + 100,
				file_path: `e2e/xy-delete-${id}.png`,
				file_type: 'image',
				is_final: true,
				created_at: created,
				width: 1024,
				height: 768
			}
		],
		...over
	};
}

const GRID = {
	id: GRID_ID,
	cols: 3,
	rows: 2,
	cell_count: 6,
	axes: [
		{ field: 'quality', label: 'Quality' },
		{ field: 'aspect', label: 'Aspect ratio' }
	]
};
const COMPARE = item('01XYDELETE00000000000001', 1, {
	grid_id: GRID_ID,
	grid_x: 0,
	grid_y: 0,
	grid_cols: 3,
	grid: GRID,
	axis_values: { quality: 'Draft', aspect: 'Square' }
});
const LOOSE = item('01XYDELETE00000000000002', 2);

const AXIS_X = {
	field: 'quality',
	type: 'select',
	label: 'Quality',
	values: ['Draft', 'Fine', 'Ultra'].map((label) => ({ value: label, label }))
};
const AXIS_Y = {
	field: 'aspect',
	type: 'select',
	label: 'Aspect ratio',
	values: ['Square', 'Wide'].map((label) => ({ value: label, label }))
};
const SERVER_GRID = {
	id: GRID_ID,
	preset_id: 'e2e/xy-delete',
	tab_id: null,
	x_axis: AXIS_X,
	y_axis: AXIS_Y,
	lock_seed: true,
	status: 'completed',
	created_at: new Date().toISOString(),
	cells: AXIS_Y.values.flatMap((row, y) =>
		AXIS_X.values.map((column, x) => ({
			x,
			y,
			generation_id: `cell-${x}-${y}`,
			status: 'completed',
			axis_values: { quality: column.label, aspect: row.label },
			seed: 1,
			thumbnail_url: `/e2e/xy-delete-cell-${x}-${y}.png`,
			media_type: 'image',
			error: null
		}))
	)
};

interface Mock {
	deletions: string[];
	failDelete: boolean;
}

async function mockHistory(page: Page, mock: Mock) {
	let removed = false;
	await page.route('**/api/generations/history**', async (route) => {
		const path = new URL(route.request().url()).pathname;
		if (path === '/api/generations/history') {
			const generations = removed ? [LOOSE] : [COMPARE, LOOSE];
			await route.fulfill({ json: { success: true, data: { generations, total: generations.length } } });
			return;
		}
		await route.fallback();
	});
	await page.route(`**/api/generations/grids/${GRID_ID}`, async (route) => {
		const method = route.request().method();
		if (method === 'GET') {
			await route.fulfill({ json: { success: true, data: SERVER_GRID } });
			return;
		}
		if (method === 'DELETE') {
			mock.deletions.push(route.request().url());
			if (mock.failDelete) {
				await route.fulfill({
					status: 500,
					json: { success: false, error: 'grid_delete_incomplete', message: 'Some images could not be deleted' }
				});
				return;
			}
			removed = true;
			await route.fulfill({ json: { success: true, data: null } });
			return;
		}
		await route.fallback();
	});
	await page.route('**/*', async (route) => {
		const request = route.request();
		if (request.resourceType() === 'image' && /xy-delete-/.test(request.url())) {
			await route.fulfill({ body: PNG_1X1, contentType: 'image/png' });
			return;
		}
		await route.fallback();
	});
}

async function openHistory(page: Page, viewport: { width: number; height: number }, mock: Mock) {
	await page.setViewportSize(viewport);
	await mockHistory(page, mock);
	await loginAsOwner(page);
	await page.goto('/history');
	await page.waitForLoadState('networkidle');
	await expect(page.locator('[data-generation-card]')).toHaveCount(2, { timeout: 15000 });
}

const deleteButton = (page: Page) => page.getByRole('button', { name: 'Delete', exact: true });

for (const viewport of [
	{ width: 1440, height: 900 },
	{ width: 390, height: 844 }
]) {
	test(`deleting an X/Y compare card removes it with one request at ${viewport.width}`, async ({ page }) => {
		const mock: Mock = { deletions: [], failDelete: false };
		await openHistory(page, viewport, mock);
		const cards = page.locator('[data-generation-card]');
		await screenshot(page, JOURNEY, `01-before-${viewport.width}`);

		await cards.first().getByRole('button', { name: 'Item actions' }).click({ force: true });
		await page.getByRole('menuitem', { name: /Delete grid/ }).click();
		await expect(page.getByText('Delete this X/Y compare? 6 images will be deleted.')).toBeVisible();
		await screenshot(page, JOURNEY, `02-confirm-${viewport.width}`);

		await deleteButton(page).click();
		await expect(cards).toHaveCount(1);
		expect(mock.deletions).toHaveLength(1);
		await screenshot(page, JOURNEY, `03-after-${viewport.width}`);
	});

	test(`a failed compare delete brings the card back with a toast at ${viewport.width}`, async ({ page }) => {
		const mock: Mock = { deletions: [], failDelete: true };
		await openHistory(page, viewport, mock);
		const cards = page.locator('[data-generation-card]');

		await cards.first().getByRole('button', { name: 'Item actions' }).click({ force: true });
		await page.getByRole('menuitem', { name: /Delete grid/ }).click();
		await deleteButton(page).click();

		await expect.poll(() => mock.deletions.length).toBe(1);
		await expect(cards).toHaveCount(2);
		await expect(page.getByRole('status').filter({ hasText: /\S/ }).first()).toBeVisible();
		await screenshot(page, JOURNEY, `04-failed-${viewport.width}`);
	});

	test(`the compare view deletes the whole compare at ${viewport.width}`, async ({ page }) => {
		const mock: Mock = { deletions: [], failDelete: false };
		await openHistory(page, viewport, mock);
		const cards = page.locator('[data-generation-card]');

		await cards.first().locator('.media-zoom').click();
		await expect(page.getByTestId('history-grid-view')).toBeVisible({ timeout: 15000 });
		await page.getByRole('button', { name: 'Delete compare' }).click();
		await expect(page.getByText('Delete this X/Y compare? 6 images will be deleted.')).toBeVisible();
		await screenshot(page, JOURNEY, `05-view-confirm-${viewport.width}`);

		await deleteButton(page).click();
		await expect(page.getByTestId('history-grid-view')).toHaveCount(0);
		await expect(cards).toHaveCount(1);
		expect(mock.deletions).toHaveLength(1);
	});
}
