import { test, expect, type Page } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot } from './helpers';
import {
	ASPECT,
	QUALITY,
	captureStartRequest,
	createGrid,
	enableFakeModels,
	openGridFromHistory,
	waitGridSettled,
	type Axis
} from './xyCompareHelpers';

const JOURNEY = 'compare-followups';

const WIDE_QUALITY: Axis = {
	...QUALITY,
	values: [1, 2, 3, 4, 5, 6, 7, 8].map((value) => ({ value, label: String(value) }))
};

const MEDIA_DELAY_MS = 1500;

async function delayMedia(page: Page) {
	await page.route('**/*', async (route) => {
		const type = route.request().resourceType();
		if (type !== 'xhr' && type !== 'fetch') return route.fallback();
		const response = await route.fetch();
		const contentType = response.headers()['content-type'] ?? '';
		if (contentType.startsWith('image/')) await new Promise((resolve) => setTimeout(resolve, MEDIA_DELAY_MS));
		await route.fulfill({ response });
	});
}

async function assertLabelsCoverPadding(page: Page) {
	const scroller = page.getByTestId('compare-scroller');
	await scroller.evaluate((el) => el.scrollTo({ left: el.scrollWidth }));
	await expect.poll(() => scroller.evaluate((el) => el.scrollLeft)).toBeGreaterThan(0);

	const covered = await scroller.evaluate((el) => {
		const box = el.getBoundingClientRect();
		const labels = [...el.querySelectorAll<HTMLElement>('[data-testid="compare-y-label"]')].filter((label) => {
			const rect = label.getBoundingClientRect();
			return rect.top >= box.top && rect.bottom <= box.bottom;
		});
		return {
			inView: labels.length,
			labelsReachScrollerEdge: labels.every((label) => label.getBoundingClientRect().left <= box.left + 1),
			stripHitsLabel: labels.every((label) => {
				const rect = label.getBoundingClientRect();
				const hit = document.elementFromPoint(box.left + 2, rect.top + rect.height / 2);
				return !!hit && label.contains(hit);
			})
		};
	});
	expect(covered.inView).toBeGreaterThan(0);
	expect(covered.labelsReachScrollerEdge).toBe(true);
	expect(covered.stripHitsLabel).toBe(true);
}

test.describe.configure({ mode: 'serial' });

test.describe('X/Y compare follow-ups', () => {
	let token = '';
	let gridId = '';

	test('setup: a settled history grid wide enough to scroll', async ({ page }) => {
		test.setTimeout(300000);
		await loginAsOwner(page);
		token = await ownerToken(page);
		await enableFakeModels(page, token);
		const request = await captureStartRequest(page);
		gridId = await createGrid(page, token, request, WIDE_QUALITY, ASPECT);
		await waitGridSettled(page, token, gridId, 180000);
	});

	test.describe('at 1440', () => {
		test.use({ viewport: { width: 1440, height: 900 } });

		test('the Stitch modal keeps its height from loading to the rendered composition', async ({ page }) => {
			test.setTimeout(240000);
			await loginAsOwner(page);
			await openGridFromHistory(page, gridId);
			await expect(page.getByTestId('compare-count')).toHaveText('16/16', { timeout: 30000 });
			await delayMedia(page);

			await page.getByTestId('compare-cell').first().click();
			await page.getByRole('button', { name: 'Export stitched' }).click();
			const dialog = page.getByRole('dialog').filter({ hasText: 'Stitch' }).last();
			await expect(dialog.getByTestId('stitch-loading')).toBeVisible({ timeout: 30000 });
			const loadingBox = (await dialog.boundingBox())!;
			await screenshot(page, JOURNEY, 'stitch-loading-1440');

			await expect(dialog.getByTestId('stitch-loading')).toHaveCount(0, { timeout: 60000 });
			await expect(dialog.getByLabel('Stitch preview', { exact: true })).toBeVisible();
			const renderedBox = (await dialog.boundingBox())!;
			expect(Math.abs(renderedBox.height - loadingBox.height)).toBeLessThanOrEqual(1);
			await screenshot(page, JOURNEY, 'stitch-rendered-1440');
		});

		test('Y labels cover the left padding strip when the grid is scrolled horizontally', async ({ page }) => {
			test.setTimeout(120000);
			await loginAsOwner(page);
			await openGridFromHistory(page, gridId);
			await expect(page.getByTestId('compare-count')).toHaveText('16/16', { timeout: 30000 });
			const scrolls = await page.getByTestId('compare-scroller').getAttribute('data-scrolls');
			if (scrolls === 'true') await assertLabelsCoverPadding(page);
			await screenshot(page, JOURNEY, 'grid-scrolled-1440');
		});
	});

	test.describe('at 390', () => {
		test.use({ viewport: { width: 390, height: 844 } });

		test('Y labels cover the left padding strip when the grid is scrolled horizontally', async ({ page }) => {
			test.setTimeout(120000);
			await loginAsOwner(page);
			await openGridFromHistory(page, gridId);
			await expect(page.getByTestId('compare-count')).toHaveText('16/16', { timeout: 30000 });
			await expect(page.getByTestId('compare-scroller')).toHaveAttribute('data-scrolls', 'true');
			await assertLabelsCoverPadding(page);
			await screenshot(page, JOURNEY, 'grid-scrolled-390');
		});

		test('the Stitch modal keeps its height from loading to the rendered composition', async ({ page }) => {
			test.setTimeout(240000);
			await loginAsOwner(page);
			await openGridFromHistory(page, gridId);
			await expect(page.getByTestId('compare-count')).toHaveText('16/16', { timeout: 30000 });
			await delayMedia(page);

			await page.getByTestId('compare-cell').first().click();
			await page.getByRole('button', { name: 'Export stitched' }).click();
			const dialog = page.getByRole('dialog').filter({ hasText: 'Stitch' }).last();
			await expect(dialog.getByTestId('stitch-loading')).toBeVisible({ timeout: 30000 });
			const loadingBox = (await dialog.boundingBox())!;
			await screenshot(page, JOURNEY, 'stitch-loading-390');

			await expect(dialog.getByTestId('stitch-loading')).toHaveCount(0, { timeout: 60000 });
			const renderedBox = (await dialog.boundingBox())!;
			expect(Math.abs(renderedBox.height - loadingBox.height)).toBeLessThanOrEqual(1);
			await screenshot(page, JOURNEY, 'stitch-rendered-390');
		});
	});
});
