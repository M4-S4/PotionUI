import { test, expect, type Page, type Route } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';
import { fixturePng } from './imageEditorHelpers';

const JOURNEY = 'media-tools-surfaces';

const PNG = fixturePng(96);
const PNG_URL = `data:image/png;base64,${PNG.toString('base64')}`;

function tool(id: string, label: string, extra: Record<string, unknown> = {}) {
	return {
		id,
		label,
		icon: 'sliders',
		accepts: ['image'],
		produces: 'same',
		cardinality: 'one_to_one',
		params: [],
		requires: {},
		selection: null,
		scopes: ['library', 'history', 'field'],
		editor: null,
		preview: false,
		component: null,
		runnable: true,
		source: 'core',
		last_params: null,
		...extra
	};
}

const CATALOG = [
	tool('trim', 'Trim', { icon: 'scissors', accepts: ['video', 'audio'], editor: 'trim' }),
	tool('crop', 'Crop', { icon: 'crop', accepts: ['image', 'video'], editor: 'crop' }),
	tool('resize', 'Resize', { icon: 'shrink', accepts: ['image', 'video'], editor: 'resize' }),
	tool('rotate', 'Rotate', {
		accepts: ['image', 'video'],
		preview: true,
		params: [{ name: 'degrees', type: 'select', label: 'Degrees', default: '90', options: ['90', '180', '270'] }]
	}),
	tool('flip', 'Flip', { accepts: ['image', 'video'] }),
	tool('speed', 'Speed', { icon: 'gauge', accepts: ['video', 'audio'] }),
	tool('fit_canvas', 'Fit canvas', { accepts: ['image'] }),
	tool('color_key', 'Color key', { accepts: ['image'] }),
	tool('crop_subject', 'Crop to subject', { accepts: ['image'], requires: { has_alpha: true } }),
	tool('frame', 'Frame', { icon: 'photo', accepts: ['video'], produces: 'image', editor: 'frame' }),
	tool('split', 'Split', { icon: 'split', accepts: ['audio'], cardinality: 'one_to_many', editor: 'split' }),
	tool('concat', 'Concat', { accepts: ['video'], cardinality: 'many_to_one', selection: { min: 2, max: 8 } })
];

function libraryRow(id: string, name: string, kind: 'image' | 'video' | 'audio') {
	return {
		id,
		filename: `${id}-file.${kind === 'image' ? 'png' : kind === 'video' ? 'mp4' : 'wav'}`,
		original_filename: name,
		media_type: kind,
		url: PNG_URL,
		thumbnail_medium: kind === 'audio' ? undefined : PNG_URL,
		width: 96,
		height: 96,
		duration_seconds: kind === 'image' ? undefined : 8,
		created_at: '2026-01-01T00:00:00Z',
		tags: [],
		is_favorite: false
	};
}

function generation(id: string, kind: 'image' | 'video', index: number) {
	return {
		id,
		preset_id: 'preset-a',
		preset_name: 'Landscape',
		status: 'completed',
		progress: 1,
		created_at: '2026-01-01T00:00:00Z',
		updated_at: '2026-01-01T00:00:00Z',
		form_data: { prompt: `prompt for ${id}` },
		files: [
			{
				id: index,
				file_path: `out/${id}/clip.${kind === 'image' ? 'png' : 'mp4'}`,
				file_type: kind,
				is_final: true,
				created_at: '2026-01-01T00:00:00Z',
				width: 96,
				height: 96,
				duration_seconds: kind === 'video' ? 8 : undefined,
				thumbnail_small: PNG_URL,
				thumbnail_medium: PNG_URL,
				thumbnail_large: PNG_URL
			}
		],
		rating: 0,
		is_favorite: false
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
	library: ReturnType<typeof libraryRow>[];
	runs: { started: unknown[]; cancelled: string[]; progress: number; finish: boolean };
}

async function mockBackend(page: Page): Promise<Backend> {
	const backend: Backend = {
		library: [
			libraryRow('lib-img', 'harbour.png', 'image'),
			libraryRow('lib-vid', 'timelapse.mp4', 'video'),
			libraryRow('lib-aud', 'ambience.wav', 'audio')
		],
		runs: { started: [], cancelled: [], progress: 0.42, finish: false }
	};
	const generations = [generation('gen-1', 'image', 1), generation('gen-2', 'video', 2)];

	await page.route(/\/api\/media-tools(\?.*)?$/, (route) => json(route, { tools: CATALOG }));
	await page.route(/\/api\/media-tools\/runs$/, async (route) => {
		backend.runs.started.push(route.request().postDataJSON());
		await json(route, { run_id: 'run-1' });
	});
	await page.route(/\/api\/media-tools\/runs\/run-1$/, async (route) => {
		if (route.request().method() === 'DELETE') {
			backend.runs.cancelled.push('run-1');
			return json(route, { cancelled: true });
		}
		if (backend.runs.finish) {
			return json(route, {
				run_id: 'run-1',
				status: 'completed',
				progress: 1,
				label: '',
				replaced: false,
				error: null,
				items: [
					{
						id: 'lib-new',
						filename: 'lib-new-file.png',
						original_filename: 'harbour-rotated.png',
						media_type: 'image',
						url: PNG_URL,
						width: 96,
						height: 96
					}
				]
			});
		}
		return json(route, {
			run_id: 'run-1',
			status: 'running',
			progress: backend.runs.progress,
			label: 'Rotating',
			items: [],
			replaced: false,
			error: null
		});
	});
	await page.route(/\/api\/media-tools\/preview$/, (route) =>
		route.fulfill({ status: 200, contentType: 'image/webp', body: PNG })
	);
	await page.route(/\/api\/library\/items(\?.*)?$/, (route) =>
		json(route, { items: backend.library, total: backend.library.length, limit: 24, offset: 0 })
	);
	await page.route(/\/api\/generations\/history(\?.*)?$/, (route) =>
		json(route, { generations, total: generations.length })
	);
	await page.route(/\/api\/tags(\?.*)?$/, (route) => json(route, { tags: [] }));
	await page.route(/\/api\/media\/generations\/gen-\d+\/.+$/, (route) =>
		route.fulfill({ status: 200, contentType: 'image/png', body: PNG })
	);
	return backend;
}

const menu = (page: Page) => page.locator('[data-tools-menu]');

async function groupLabels(page: Page): Promise<string[]> {
	return page
		.locator('[data-tools-menu] [role="group"]')
		.evaluateAll((groups) => groups.map((group) => group.getAttribute('aria-label') ?? ''));
}

async function toolsOf(page: Page, group: string): Promise<string[]> {
	return page
		.locator(`[data-tools-menu] [role="group"][aria-label="${group}"] [data-tool]`)
		.evaluateAll((rows) => rows.map((row) => row.getAttribute('data-tool') ?? ''));
}

async function openLibraryMenu(page: Page, name: string) {
	await page.goto('/library');
	await expect(page.locator('[data-library-card]')).toHaveCount(3);
	await page.locator('[data-library-card]').filter({ hasText: name }).getByRole('button', { name: 'Select item' }).click();
	await page.locator('[data-tools-trigger]').first().click();
	await expect(menu(page)).toBeVisible();
}

async function openHistoryMenu(page: Page, index: number) {
	await page.goto('/history');
	const selectors = page.getByRole('button', { name: 'Select generation' });
	await expect(selectors).toHaveCount(2);
	await selectors.nth(index).click();
	await page.locator('[data-tools-trigger]').first().click();
	await expect(menu(page)).toBeVisible();
}

test.describe('Tools menu, Edit and Effects', () => {
	test.beforeEach(async ({ page }) => {
		await page.setViewportSize({ width: 1440, height: 900 });
		await loginAsOwner(page);
		await mockBackend(page);
	});

	test('Library: an image gets Edit then Effects ahead of the existing groups', async ({ page }) => {
		await openLibraryMenu(page, 'harbour.png');

		const labels = await groupLabels(page);
		expect(labels.slice(0, 2)).toEqual(['Edit', 'Effects']);
		expect(labels).toEqual(expect.arrayContaining(['Analyze', 'Compose', 'Export']));
		expect(await toolsOf(page, 'Edit')).toEqual([
			'crop',
			'resize',
			'rotate',
			'flip',
			'fit_canvas',
			'color_key',
			'crop_subject'
		]);
		expect(await toolsOf(page, 'Effects')).toEqual(['apply-look']);
		await expect(menu(page).locator('[data-tool="crop_subject"]')).toHaveAttribute('aria-disabled', 'true');
		await expect(menu(page).locator('[data-tool="crop_subject"] [data-tool-sub]')).toHaveText(
			'Needs a transparent image'
		);
		await screenshot(page, JOURNEY, 'library-image-menu-1440');
	});

	test('Library: a video gets the video tools and no Effects, with Concat waiting for a second clip', async ({ page }) => {
		await openLibraryMenu(page, 'timelapse.mp4');

		expect(await toolsOf(page, 'Edit')).toEqual(['trim', 'crop', 'resize', 'rotate', 'flip', 'speed', 'frame', 'concat']);
		expect(await groupLabels(page)).not.toContain('Effects');
		await expect(menu(page).locator('[data-tool="concat"] [data-tool-sub]')).toHaveText('Select 2 to 8 items');
		await expect(menu(page).locator('[data-tool="frame"] [data-tool-sub]')).toHaveText('Saves one image');
		await screenshot(page, JOURNEY, 'library-video-menu-1440');
	});

	test('Library: audio gets trim, speed and split and no Effects', async ({ page }) => {
		await openLibraryMenu(page, 'ambience.wav');

		expect(await toolsOf(page, 'Edit')).toEqual(['trim', 'speed', 'split']);
		expect(await groupLabels(page)).not.toContain('Effects');
		await screenshot(page, JOURNEY, 'library-audio-menu-1440');
	});

	test('History shows the same Edit and Effects groups as Library', async ({ page }) => {
		await openHistoryMenu(page, 0);
		const historyImage = { edit: await toolsOf(page, 'Edit'), effects: await toolsOf(page, 'Effects') };
		await page.keyboard.press('Escape');

		await openLibraryMenu(page, 'harbour.png');
		expect(historyImage.edit).toEqual(await toolsOf(page, 'Edit'));
		expect(historyImage.effects).toEqual(await toolsOf(page, 'Effects'));

		await openHistoryMenu(page, 1);
		expect(await toolsOf(page, 'Edit')).toEqual(['trim', 'crop', 'resize', 'rotate', 'flip', 'speed', 'frame', 'concat']);
		expect(await groupLabels(page)).not.toContain('Effects');
		await screenshot(page, JOURNEY, 'history-video-menu-1440');
	});

	test('Apply a look opens the image editor on the Filters tool', async ({ page }) => {
		test.setTimeout(120000);
		await openLibraryMenu(page, 'harbour.png');

		await menu(page).locator('[data-tool="apply-look"]').click();

		const dialog = page.getByRole('dialog', { name: 'Edit image' });
		await expect(dialog).toBeVisible();
		await expect(dialog.getByRole('button', { name: 'Filters', exact: true }).first()).toHaveAttribute(
			'aria-pressed',
			'true',
			{ timeout: 20000 }
		);
		await expect(dialog.getByRole('radiogroup', { name: 'Filters' })).toBeVisible();
		await screenshot(page, JOURNEY, 'apply-look-1440');
	});
});

test.describe('recent looks', () => {
	test.beforeEach(async ({ page }) => {
		await page.setViewportSize({ width: 1440, height: 900 });
		await loginAsOwner(page);
	});

	test('lists the looks applied last under Apply a look, on Library and History', async ({ page }) => {
		await page.addInitScript(() => {
			localStorage.setItem(
				'potionui-recent-looks',
				JSON.stringify([
					{ id: 'noir', name: 'Noir', group: 'Black and white' },
					{ id: 'ember', name: 'Ember', group: 'Colour' },
					{ id: 'honey', name: 'Honey', group: 'Colour' },
					{ id: 'frost', name: 'Frost', group: 'Colour' }
				])
			);
		});
		await mockBackend(page);

		await openLibraryMenu(page, 'harbour.png');
		expect(await toolsOf(page, 'Effects')).toEqual(['apply-look', 'look:noir', 'look:ember', 'look:honey']);
		await expect(menu(page).locator('[data-tool="look:noir"] [data-tool-sub]')).toHaveText('Black and white');
		await screenshot(page, JOURNEY, 'recent-looks-menu-1440');
		await page.keyboard.press('Escape');

		await openHistoryMenu(page, 0);
		expect(await toolsOf(page, 'Effects')).toEqual(['apply-look', 'look:noir', 'look:ember', 'look:honey']);
	});

	test('opens the image editor on the look that was clicked', async ({ page }) => {
		test.setTimeout(120000);
		await page.addInitScript(() => {
			localStorage.setItem(
				'potionui-recent-looks',
				JSON.stringify([{ id: 'ember', name: 'Ember', group: 'Colour' }])
			);
		});
		await mockBackend(page);
		await openLibraryMenu(page, 'harbour.png');

		await menu(page).locator('[data-tool="look:ember"]').click();

		const dialog = page.getByRole('dialog', { name: 'Edit image' });
		await expect(dialog).toBeVisible();
		await expect(dialog.getByRole('button', { name: 'Filters', exact: true }).first()).toHaveAttribute(
			'aria-pressed',
			'true',
			{ timeout: 20000 }
		);
		await expect(dialog.getByTestId('filter-name')).toHaveText('Ember', { timeout: 20000 });
		await expect(dialog.getByRole('button', { name: 'Apply', exact: true })).toBeFocused();
		await screenshot(page, JOURNEY, 'recent-look-editor-1440');
	});

	test('still renders the menu when the browser refuses storage', async ({ page }) => {
		await page.addInitScript(() => {
			Object.defineProperty(window, 'localStorage', {
				get() {
					throw new DOMException('blocked', 'SecurityError');
				}
			});
		});
		await mockBackend(page);
		await openLibraryMenu(page, 'harbour.png');

		expect(await toolsOf(page, 'Effects')).toEqual(['apply-look']);
	});
});

test.describe('the tool catalog fails to load', () => {
	test.beforeEach(async ({ page }) => {
		await page.setViewportSize({ width: 1440, height: 900 });
		await loginAsOwner(page);
	});

	async function mockCatalog(page: Page, state: { healthy: boolean; requests: number }) {
		await mockBackend(page);
		await page.route(/\/api\/media-tools(\?.*)?$/, async (route) => {
			state.requests += 1;
			if (!state.healthy) {
				return route.fulfill({ status: 500, contentType: 'application/json', body: '{"success":false}' });
			}
			return json(route, { tools: CATALOG });
		});
	}

	async function selectHarbour(page: Page) {
		await page.goto('/library');
		await expect(page.locator('[data-library-card]')).toHaveCount(3);
		await page
			.locator('[data-library-card]')
			.filter({ hasText: 'harbour.png' })
			.getByRole('button', { name: 'Select item' })
			.click();
	}

	test('shows one toast, and the next menu open brings the Edit group back', async ({ page }) => {
		const state = { healthy: false, requests: 0 };
		await mockCatalog(page, state);
		await selectHarbour(page);
		await expect(page.getByText("Editing tools couldn't load")).toHaveCount(1);

		state.healthy = true;
		await page.locator('[data-tools-trigger]').first().click();

		await expect.poll(() => groupLabels(page)).toContain('Edit');
		expect(state.requests).toBe(2);
		await expect(page.getByText("Editing tools couldn't load")).toHaveCount(1);
		await screenshot(page, JOURNEY, 'catalog-retry-1440');
	});

	test('retries once only, and stays quiet while the catalog is still down', async ({ page }) => {
		const state = { healthy: false, requests: 0 };
		await mockCatalog(page, state);
		await selectHarbour(page);
		await expect(page.getByText("Editing tools couldn't load")).toHaveCount(1);

		for (let attempt = 0; attempt < 3; attempt += 1) {
			await page.locator('[data-tools-trigger]').first().click();
			await expect(menu(page)).toBeVisible();
			await page.keyboard.press('Escape');
		}

		expect(state.requests).toBe(2);
		await expect(page.getByText("Editing tools couldn't load")).toHaveCount(1);
	});
});

test.describe('card menu', () => {
	test.beforeEach(async ({ page }) => {
		await page.setViewportSize({ width: 1440, height: 900 });
		await loginAsOwner(page);
		await mockBackend(page);
	});

	test('starts with the same Edit group on a Library card and a History card', async ({ page }) => {
		await page.goto('/library');
		const libraryCard = page.locator('[data-library-card]').filter({ hasText: 'timelapse.mp4' });
		await libraryCard.hover();
		await libraryCard.getByRole('button', { name: 'Item actions' }).click();
		const libraryGroups = await groupLabels(page);
		const libraryEdit = await toolsOf(page, 'Edit');
		await page.keyboard.press('Escape');

		await page.goto('/history');
		const historyCard = page.locator('[data-generation-card]').nth(1);
		await historyCard.hover();
		await historyCard.getByRole('button', { name: 'Item actions' }).click();

		expect(libraryGroups.slice(0, 1)).toEqual(['Edit']);
		expect((await groupLabels(page)).slice(0, 1)).toEqual(['Edit']);
		expect(await toolsOf(page, 'Edit')).toEqual(libraryEdit);
		expect(libraryEdit).not.toContain('concat');
		await screenshot(page, JOURNEY, 'history-card-menu-1440');
	});

	test('a Library image card offers Apply a look under Effects and a video card does not', async ({ page }) => {
		await page.goto('/library');
		const image = page.locator('[data-library-card]').filter({ hasText: 'harbour.png' });
		await image.hover();
		await image.getByRole('button', { name: 'Item actions' }).click();
		expect(await toolsOf(page, 'Effects')).toEqual(['apply-look']);
		await page.keyboard.press('Escape');

		const video = page.locator('[data-library-card]').filter({ hasText: 'timelapse.mp4' });
		await video.hover();
		await video.getByRole('button', { name: 'Item actions' }).click();
		expect(await groupLabels(page)).not.toContain('Effects');
	});
});

test.describe('details rows', () => {
	test.beforeEach(async ({ page }) => {
		await page.setViewportSize({ width: 1440, height: 900 });
		await loginAsOwner(page);
		await mockBackend(page);
	});

	test('Library details: Edit, Effects and the Library actions in their own group', async ({ page }) => {
		await page.goto('/library');
		await page.locator('[data-library-card]').filter({ hasText: 'harbour.png' }).locator('.media-zoom').click();

		const row = page.locator('[data-media-tool-row]');
		await expect(row).toBeVisible();
		await expect(row.locator('[data-row-group="edit"] button')).toHaveText([
			'Crop',
			'Resize',
			'Edit image',
			'More tools'
		]);
		await expect(row.locator('[data-row-group="effects"] button')).toHaveText(['Apply a look…']);
		await expect(row.locator('[data-row-group="page"]')).toHaveAttribute('aria-label', 'Library actions');
		await expect(row.locator('[data-row-group="page"] button')).toHaveText([
			'Auto-file items like this…',
			'Delete',
			'Download'
		]);
		await expect(page.locator('[data-library-favorite]')).toBeVisible();
		await screenshot(page, JOURNEY, 'library-details-1440');
	});

	test('Library details: More tools holds the rest', async ({ page }) => {
		await page.goto('/library');
		await page.locator('[data-library-card]').filter({ hasText: 'harbour.png' }).locator('.media-zoom').click();

		await page.locator('[data-media-tool-row]').getByRole('button', { name: 'More tools' }).click();
		const more = page.getByRole('menu', { name: 'More tools' });
		await expect(more.locator('[data-row-tool]')).toHaveCount(5);
		await expect(more.locator('[data-row-tool="crop_subject"]')).toHaveAttribute('aria-disabled', 'true');
		await screenshot(page, JOURNEY, 'library-details-more-1440');
	});

	test('History details: the same Edit and Effects, the generation actions in their own group', async ({ page }) => {
		await page.goto('/history');
		await page.locator('[data-generation-card]').first().locator('.media-zoom').click();

		const dialog = page.getByRole('dialog', { name: 'Generation Details' });
		const row = dialog.locator('[data-media-tool-row]');
		await expect(row).toBeVisible();
		await expect(row.locator('[data-row-group="edit"] button')).toHaveText([
			'Crop',
			'Resize',
			'Edit image',
			'More tools'
		]);
		await expect(row.locator('[data-row-group="effects"] button')).toHaveText(['Apply a look…']);
		await expect(row.locator('[data-row-group="page"]')).toHaveAttribute('aria-label', 'Generation actions');
		await expect(row.locator('[data-row-group="page"] button')).toHaveText([
			'Reuse settings',
			'Copy to Library',
			'Download'
		]);
		const corner = dialog.locator('[data-stage-toolbar]');
		await expect(corner.getByRole('button', { name: 'Add to favorites' })).toBeAttached();
		await expect(corner.getByRole('button', { name: 'Publish to Inspirations' })).toBeAttached();
		await expect(corner.getByRole('button', { name: 'Display in new tab' })).toBeAttached();
		await expect(corner.getByRole('button', { name: 'Download' })).toHaveCount(0);
		await screenshot(page, JOURNEY, 'history-details-1440');
	});

	test('History details: a video generation can be trimmed, cropped and resized', async ({ page }) => {
		await page.goto('/history');
		await page.locator('[data-generation-card]').nth(1).locator('.media-zoom').click();

		const row = page.getByRole('dialog', { name: 'Generation Details' }).locator('[data-media-tool-row]');
		await expect(row.locator('[data-row-group="edit"] button')).toHaveText([
			'Trim',
			'Crop',
			'Resize',
			'Frame',
			'More tools'
		]);
		await expect(row.locator('[data-row-group="effects"]')).toHaveCount(0);
	});
});

test.describe('running a tool', () => {
	test.beforeEach(async ({ page }) => {
		await page.setViewportSize({ width: 1440, height: 900 });
		await loginAsOwner(page);
	});

	test('shows progress on the card with a cancel button, then the result next to the source tagged New', async ({
		page
	}) => {
		test.setTimeout(120000);
		const backend = await mockBackend(page);
		await page.goto('/library');
		const card = page.locator('[data-library-card]').filter({ hasText: 'harbour.png' });
		await card.hover();
		await card.getByRole('button', { name: 'Item actions' }).click();
		await menu(page).locator('[data-tool="rotate"]').click();

		const dialog = page.getByRole('dialog', { name: /Rotate/ });
		await expect(dialog).toBeVisible();
		await dialog.getByRole('button', { name: 'Apply' }).click();

		const band = card.locator('[data-tool-run]');
		await expect(band).toBeVisible();
		await expect(band.locator('[data-tool-run-percent]')).toHaveText('42%');
		await expect(band.getByRole('button', { name: /Cancel/ })).toBeVisible();
		expect(backend.runs.started).toHaveLength(1);
		await screenshot(page, JOURNEY, 'running-1440');

		backend.library = [libraryRowAfter(), ...backend.library];
		backend.runs.finish = true;
		await expect(band).toHaveCount(0, { timeout: 15000 });

		const added = page.locator('[data-library-card="lib-new"]');
		await expect(added).toBeVisible();
		await expect(added.locator('[data-new-tag]')).toHaveText('New');
		const order = await page
			.locator('[data-library-card]')
			.evaluateAll((cards) => cards.map((node) => node.getAttribute('data-library-card')));
		expect(order.indexOf('lib-new')).toBe(order.indexOf('lib-img') - 1);
		await screenshot(page, JOURNEY, 'finished-1440');
	});

	test('cancelling from the card stops the run', async ({ page }) => {
		const backend = await mockBackend(page);
		await page.goto('/library');
		const card = page.locator('[data-library-card]').filter({ hasText: 'harbour.png' });
		await card.hover();
		await card.getByRole('button', { name: 'Item actions' }).click();
		await menu(page).locator('[data-tool="rotate"]').click();
		await page.getByRole('dialog', { name: /Rotate/ }).getByRole('button', { name: 'Apply' }).click();

		await card.locator('[data-tool-run]').getByRole('button', { name: /Cancel/ }).click();

		await expect.poll(() => backend.runs.cancelled).toEqual(['run-1']);
		await expect(card.locator('[data-tool-run]')).toHaveCount(0);
	});

	test('a run started from History saves to the Library and links to it', async ({ page }) => {
		test.setTimeout(120000);
		const backend = await mockBackend(page);
		await page.goto('/history');
		const card = page.locator('[data-generation-card]').first();
		await card.hover();
		await card.getByRole('button', { name: 'Item actions' }).click();
		await menu(page).locator('[data-tool="rotate"]').click();
		const dialog = page.getByRole('dialog', { name: /Rotate/ });
		await expect(dialog).toContainText('Saved as a new Library item');
		await dialog.getByRole('button', { name: 'Apply' }).click();

		await expect(card.locator('[data-tool-run]')).toBeVisible();
		backend.runs.finish = true;

		const toast = page.getByRole('status').filter({ hasText: 'Saved to Library' });
		await expect(toast).toBeVisible({ timeout: 15000 });
		await toast.getByRole('button', { name: 'Open Library' }).click();
		await expect(page).toHaveURL(/\/library/);
	});
});

function libraryRowAfter() {
	return {
		...libraryRow('lib-new', 'harbour-rotated.png', 'image'),
		filename: 'lib-new-file.png',
		url: PNG_URL,
		created_at: new Date().toISOString(),
		tags: [],
		is_favorite: false
	};
}
