import { test, expect, type Page, type Route } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';

const JOURNEY = 'library-favorite';

const PNG = Buffer.from(
	'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
	'base64'
);
const PNG_URL = `data:image/png;base64,${PNG.toString('base64')}`;

interface Row {
	id: string;
	filename: string;
	original_filename: string;
	media_type: string;
	url: string;
	thumbnail_medium: string;
	width: number;
	height: number;
	tags: never[];
	is_favorite: boolean;
	created_at: string;
}

function row(id: string, name: string, favorite = false): Row {
	return {
		id,
		filename: `${id}-file.png`,
		original_filename: name,
		media_type: 'image',
		url: PNG_URL,
		thumbnail_medium: PNG_URL,
		width: 96,
		height: 96,
		tags: [],
		is_favorite: favorite,
		created_at: '2026-01-01T00:00:00Z'
	};
}

async function json(route: Route, data: unknown) {
	await route.fulfill({
		status: 200,
		contentType: 'application/json',
		body: JSON.stringify({ success: true, data })
	});
}

interface Backend {
	rows: Row[];
	writes: Array<{ id: string; is_favorite: boolean }>;
}

async function mockLibrary(page: Page): Promise<Backend> {
	const backend: Backend = {
		rows: [row('lib-1', 'harbour.png'), row('lib-2', 'dunes.png'), row('lib-3', 'forest.png', true)],
		writes: []
	};

	await page.route(/\/api\/library\/items(\?.*)?$/, (route) => {
		const url = new URL(route.request().url());
		const favoritesOnly = url.searchParams.get('favorites_only') === 'true';
		const items = favoritesOnly ? backend.rows.filter((entry) => entry.is_favorite) : backend.rows;
		return json(route, { items, total: items.length, limit: 24, offset: 0 });
	});
	await page.route(/\/api\/library\/items\/[^/]+\/favorite$/, async (route) => {
		const id = route.request().url().split('/').slice(-2)[0];
		const body = route.request().postDataJSON() as { is_favorite: boolean };
		backend.writes.push({ id, is_favorite: body.is_favorite });
		backend.rows = backend.rows.map((entry) =>
			entry.id === id ? { ...entry, is_favorite: body.is_favorite } : entry
		);
		await json(route, { id, is_favorite: body.is_favorite });
	});
	await page.route(/\/api\/collections\/?(\?.*)?$/, (route) =>
		json(route, {
			collections: [],
			smart_counts: {
				all: backend.rows.length,
				favorites: backend.rows.filter((entry) => entry.is_favorite).length,
				unsorted: backend.rows.length
			}
		})
	);
	await page.route(/\/api\/library\/facets$/, (route) =>
		json(route, { media_types: { image: backend.rows.length } })
	);
	await page.route(/\/api\/tags(\?.*)?$/, (route) => json(route, { tags: [] }));
	return backend;
}

test.describe('Library favorite', () => {
	test.beforeEach(async ({ page }) => {
		await page.setViewportSize({ width: 1440, height: 900 });
		await loginAsOwner(page);
	});

	test('favorites an item from its card menu and reads Remove favorite afterwards', async ({ page }) => {
		const backend = await mockLibrary(page);
		await page.goto('/library');
		const cards = page.locator('[data-library-card]');
		await expect(cards).toHaveCount(3);

		const card = cards.filter({ hasText: 'harbour.png' });
		await card.hover();
		await card.getByRole('button', { name: 'Item actions' }).click();
		const row = page.locator('[data-tools-menu] [data-tool="favorite"]');
		await expect(row).toContainText('Favorite');
		await screenshot(page, JOURNEY, 'card-menu-1440');
		await row.click();

		await expect.poll(() => backend.writes).toEqual([{ id: 'lib-1', is_favorite: true }]);

		await card.hover();
		await card.getByRole('button', { name: 'Item actions' }).click();
		await expect(page.locator('[data-tools-menu] [data-tool="favorite"]')).toContainText(
			'Remove favorite'
		);
	});

	test('favorites from the details corner and shows the heart filled', async ({ page }) => {
		const backend = await mockLibrary(page);
		await page.goto('/library');
		await page.locator('[data-library-card]').filter({ hasText: 'dunes.png' }).locator('.media-zoom').click();

		const corner = page.locator('[data-library-favorite]');
		await expect(corner).toBeVisible();
		await corner.getByRole('button', { name: 'Add to favorites' }).click();

		await expect(corner.getByRole('button', { name: 'Remove from favorites' })).toBeVisible();
		await expect.poll(() => backend.writes).toEqual([{ id: 'lib-2', is_favorite: true }]);
		await screenshot(page, JOURNEY, 'details-corner-1440');
	});

	test('the Favorites view and its count follow a change made on the page', async ({ page }) => {
		const backend = await mockLibrary(page);
		await page.goto('/library');
		await expect(page.locator('[data-library-card]')).toHaveCount(3);

		const favorites = page.getByRole('button', { name: /Favorites/ });
		await expect(favorites).toContainText('1');

		const card = page.locator('[data-library-card]').filter({ hasText: 'harbour.png' });
		await card.hover();
		await card.getByRole('button', { name: 'Item actions' }).click();
		await page.locator('[data-tools-menu] [data-tool="favorite"]').click();
		await expect.poll(() => backend.writes.length).toBe(1);
		await expect(favorites).toContainText('2');

		await favorites.click();
		await expect(page.locator('[data-library-card]')).toHaveCount(2);

		const favorite = page.locator('[data-library-card]').filter({ hasText: 'forest.png' });
		await favorite.hover();
		await favorite.getByRole('button', { name: 'Item actions' }).click();
		await expect(page.locator('[data-tools-menu] [data-tool="favorite"]')).toContainText(
			'Remove favorite'
		);
		await page.locator('[data-tools-menu] [data-tool="favorite"]').click();

		await expect(page.locator('[data-library-card]')).toHaveCount(1);
		await expect(favorites).toContainText('1');
	});
});
