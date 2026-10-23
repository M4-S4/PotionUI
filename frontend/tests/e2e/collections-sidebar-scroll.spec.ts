import { test, expect, type Page } from '@playwright/test';
import { fulfillJson, loginAsOwner } from './helpers';

function historyCollections(count: number) {
	return Array.from({ length: count }, (_, index) => ({
		id: `e2e-collection-${index}`,
		name: `E2E collection ${String(index + 1).padStart(2, '0')}`,
		scope: 'history',
		parent_id: null,
		created_at: '2026-01-01T00:00:00Z',
		item_count: index
	}));
}

async function serveHistoryCollections(page: Page, count: number) {
	await page.route(
		(url) => url.pathname === '/api/collections' && url.searchParams.get('scope') === 'history',
		(route) => {
			if (route.request().method() !== 'GET') return route.fallback();
			return fulfillJson(route, {
				success: true,
				data: {
					collections: historyCollections(count),
					total: count,
					smart_counts: { all: count, favorites: 0, unsorted: 0 }
				}
			});
		}
	);
}

async function treeOverflow(page: Page, rows: number) {
	await page.goto('/history');
	const tree = page.locator('aside [role="tree"][aria-label="Collections"]');
	await expect(tree.getByRole('treeitem')).toHaveCount(rows, { timeout: 15000 });
	return tree.evaluate((el) => ({ scrollHeight: el.scrollHeight, clientHeight: el.clientHeight }));
}

test.describe('collections sidebar scrolling', () => {
	test.beforeEach(async ({ page }) => {
		await page.setViewportSize({ width: 1440, height: 900 });
		await loginAsOwner(page);
	});

	test('has nothing to scroll with two collections', async ({ page }) => {
		await serveHistoryCollections(page, 2);
		const { scrollHeight, clientHeight } = await treeOverflow(page, 2);
		expect(clientHeight).toBeGreaterThan(0);
		expect(scrollHeight).toBeLessThanOrEqual(clientHeight);
	});

	test('scrolls the tree with forty collections', async ({ page }) => {
		await serveHistoryCollections(page, 40);
		const { scrollHeight, clientHeight } = await treeOverflow(page, 40);
		expect(scrollHeight).toBeGreaterThan(clientHeight);
	});
});
