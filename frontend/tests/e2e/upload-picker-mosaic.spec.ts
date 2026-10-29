import { test, expect, type Page, type Route } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';

const JOURNEY = 'upload-picker-mosaic';

const VIEWPORTS = [
	{ name: '1440', width: 1440, height: 900 },
	{ name: '390', width: 390, height: 844 }
];

function swatch(color: string, width: number, height: number): string {
	const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="${width}" height="${height}" fill="${color}"/><circle cx="${width / 2}" cy="${height / 2}" r="${Math.min(width, height) / 4}" fill="#ffffff" fill-opacity="0.25"/></svg>`;
	return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

function image(id: string, name: string, color: string, width: number, height: number) {
	return {
		id,
		filename: `stored-${id}`,
		original_filename: name,
		media_type: 'image',
		url: swatch(color, width, height),
		thumbnail_medium: swatch(color, Math.round(width / 4), Math.round(height / 4)),
		width,
		height,
		size: width * height,
		created_at: '2026-01-01T00:00:00Z',
		tags: []
	};
}

const LIBRARY = [
	image('lib-portrait', 'lighthouse.png', '#6b7a8f', 832, 1216),
	image('lib-square', 'teapot.png', '#8f7a6b', 1024, 1024),
	image('lib-landscape', 'salt-flats.png', '#5f8f7a', 1536, 864),
	image('lib-wide', 'harbour.png', '#7a5f8f', 1536, 864),
	{
		id: 'lib-video',
		filename: 'stored-video',
		original_filename: 'walkthrough.mp4',
		media_type: 'video',
		url: '',
		thumbnail_medium: swatch('#4f6f5f', 320, 180),
		width: 1280,
		height: 720,
		duration_seconds: 8.4,
		fps: 24,
		size: 3400000,
		created_at: '2026-01-01T00:00:00Z',
		tags: []
	},
	{
		id: 'lib-video-bare',
		filename: 'stored-video-bare',
		original_filename: 'clip-without-thumbnail.mp4',
		media_type: 'video',
		url: '',
		created_at: '2026-01-01T00:00:00Z',
		tags: []
	},
	{
		id: 'lib-audio',
		filename: 'stored-audio',
		original_filename: 'rain.wav',
		media_type: 'audio',
		url: '',
		duration_seconds: 12,
		size: 480000,
		created_at: '2026-01-01T00:00:00Z',
		tags: []
	}
];

async function json(route: Route, data: unknown) {
	await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ success: true, data }) });
}

async function mockMedia(page: Page) {
	await page.route(/\/api\/library\/items(\?.*)?$/, (route) =>
		json(route, { items: LIBRARY, total: LIBRARY.length, limit: 20, offset: 0 })
	);
	await page.route(/\/api\/library\/items\/lib-square$/, (route) =>
		json(route, { id: 'lib-square', deleted: true })
	);
	await page.route(/\/api\/tags(\?.*)?$/, (route) => json(route, { tags: [] }));
}

async function openPicker(page: Page, props: Record<string, unknown>) {
	await page.evaluate((props) => {
		const entry = (window as any).__potionui.components.UploadLibraryModal;
		const el = document.body.appendChild(document.createElement('div'));
		(window as any).__pickerEvents = { selected: null, confirmed: null };
		const handle = entry.mount(el, {
			isOpen: true,
			...props,
			onSelect: (item: { id: string }) => ((window as any).__pickerEvents.selected = item.id),
			onConfirm: (keys: string[]) => ((window as any).__pickerEvents.confirmed = keys),
			onClose: () => {
				entry.unmount(handle);
				el.remove();
			}
		});
	}, props);
}

async function box(page: Page, selector: string) {
	const found = await page.locator(selector).boundingBox();
	if (!found) throw new Error(`no box for ${selector}`);
	return found;
}

async function frameBox(page: Page, itemId: string) {
	return box(page, `[data-item-id="${itemId}"] .media-zoom`);
}

async function borderColorOfClass(page: Page, className: string): Promise<string> {
	return page.evaluate((className) => {
		const probe = document.createElement('div');
		probe.className = `${className} border`;
		document.body.appendChild(probe);
		const color = getComputedStyle(probe).borderTopColor;
		probe.remove();
		return color;
	}, className);
}

async function rows(page: Page) {
	const tiles = await page.locator('[data-testid="library-file-tile"]').evaluateAll((els) =>
		els.map((el) => {
			const rect = el.getBoundingClientRect();
			return { id: el.getAttribute('data-item-id')!, top: Math.round(rect.top), left: rect.left, width: rect.width };
		})
	);
	const grouped = new Map<number, typeof tiles>();
	for (const tile of tiles) {
		const key = [...grouped.keys()].find((top) => Math.abs(top - tile.top) <= 2) ?? tile.top;
		grouped.set(key, [...(grouped.get(key) ?? []), tile]);
	}
	return [...grouped.values()];
}

for (const viewport of VIEWPORTS) {
	test(`upload picker mosaic single-pick at ${viewport.name}`, async ({ page }) => {
		await page.setViewportSize({ width: viewport.width, height: viewport.height });
		await mockMedia(page);
		await loginAsOwner(page);
		await page.waitForFunction(() => !!(window as any).__potionui?.components?.UploadLibraryModal);

		await openPicker(page, { multiple: false });
		const dialog = page.getByRole('dialog');
		await expect(dialog.locator('[data-item-id="lib-portrait"]')).toBeVisible({ timeout: 20000 });
		await expect(dialog.locator('[data-testid="library-file-tile"]')).toHaveCount(7);

		const portrait = await frameBox(page, 'lib-portrait');
		const square = await frameBox(page, 'lib-square');
		const landscape = await frameBox(page, 'lib-landscape');
		const video = await frameBox(page, 'lib-video');
		const bareVideo = await frameBox(page, 'lib-video-bare');
		const audio = await frameBox(page, 'lib-audio');
		expect(portrait.width / portrait.height).toBeCloseTo(832 / 1216, 1);
		expect(square.width / square.height).toBeCloseTo(1, 1);
		expect(landscape.width / landscape.height).toBeCloseTo(1536 / 864, 1);
		expect(video.width / video.height).toBeCloseTo(1280 / 720, 1);
		expect(bareVideo.width / bareVideo.height).toBeCloseTo(1, 1);
		expect(audio.width / audio.height).toBeCloseTo(1, 1);

		for (const id of ['lib-portrait', 'lib-landscape']) {
			const img = dialog.locator(`[data-item-id="${id}"] .media-zoom img`);
			await expect(img).toHaveCSS('object-fit', 'contain');
			const frame = await frameBox(page, id);
			const picture = await box(page, `[data-item-id="${id}"] .media-zoom img`);
			expect(Math.abs(picture.width - frame.width)).toBeLessThan(1.5);
			expect(Math.abs(picture.height - frame.height)).toBeLessThan(1.5);
		}

		const mosaicWidth = (await box(page, '[data-testid="library-mosaic"]')).width;
		const mosaicRows = await rows(page);
		expect(mosaicRows.length).toBeGreaterThan(1);
		for (const row of mosaicRows.slice(0, -1)) {
			const filled = row.reduce((sum, tile) => sum + tile.width, 0) + 12 * (row.length - 1);
			expect(Math.abs(filled - mosaicWidth)).toBeLessThan(4);
		}

		await screenshot(page, JOURNEY, `single-${viewport.name}`);

		const tileButton = dialog.getByRole('button', { name: 'Use salt-flats.png' });
		const tileImage = tileButton.locator('.media-zoom img').first();
		await expect(tileImage).toHaveCSS('transform', 'none');
		await tileButton.hover();
		await expect(tileImage).not.toHaveCSS('transform', 'none');
		await expect(tileButton.locator('[data-library-card-body] > div').first()).toHaveCSS(
			'border-top-color',
			await borderColorOfClass(page, 'border-line-hover')
		);

		await tileButton.click();
		expect(await page.evaluate(() => (window as any).__pickerEvents.selected)).toBe('lib-landscape');
	});

	test(`upload picker mosaic single-pick delete at ${viewport.name}`, async ({ page }) => {
		await page.setViewportSize({ width: viewport.width, height: viewport.height });
		await mockMedia(page);
		await loginAsOwner(page);
		await page.waitForFunction(() => !!(window as any).__potionui?.components?.UploadLibraryModal);

		await openPicker(page, { multiple: false });
		const dialog = page.getByRole('dialog');
		await expect(dialog.locator('[data-item-id="lib-square"]')).toBeVisible({ timeout: 20000 });

		const trash = dialog.getByRole('button', { name: 'Delete teapot.png' });
		await trash.click();
		await expect(dialog.getByRole('button', { name: 'Delete', exact: true })).toBeVisible();
		await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
		await expect(dialog.getByRole('button', { name: 'Delete', exact: true })).toHaveCount(0);
		await expect(dialog.locator('[data-item-id="lib-square"]')).toHaveCount(1);

		await trash.click();
		await screenshot(page, JOURNEY, `single-delete-${viewport.name}`);
		await dialog.getByRole('button', { name: 'Delete', exact: true }).click();
		await expect(dialog.locator('[data-item-id="lib-square"]')).toHaveCount(0);
		await expect(dialog.locator('[data-testid="library-file-tile"]')).toHaveCount(6);
		expect(await page.evaluate(() => (window as any).__pickerEvents.selected)).toBeNull();
	});

	test(`upload picker mosaic multi-select at ${viewport.name}`, async ({ page }) => {
		await page.setViewportSize({ width: viewport.width, height: viewport.height });
		await mockMedia(page);
		await loginAsOwner(page);
		await page.waitForFunction(() => !!(window as any).__potionui?.components?.UploadLibraryModal);

		await openPicker(page, { multiple: true, selectableMediaTypes: ['image', 'video'] });
		const dialog = page.getByRole('dialog');
		await expect(dialog.locator('[data-item-id="lib-portrait"]')).toBeVisible({ timeout: 20000 });
		await expect(dialog.locator('[data-testid="library-file-tile"]')).toHaveCount(7);

		await dialog.getByRole('checkbox', { name: 'Select lighthouse.png' }).check();
		await dialog.getByRole('checkbox', { name: 'Select salt-flats.png' }).check();
		await expect(dialog.getByTestId('picker-selected-count')).toHaveText('2 selected');
		await expect(dialog.locator('[data-item-id="lib-portrait"]')).toHaveAttribute('data-selected', 'true');
		await expect(dialog.getByRole('checkbox', { name: 'Select rain.wav' })).toBeDisabled();
		await expect(dialog.getByText('Only image and video files can be selected here').first()).toBeVisible();

		await dialog.locator('[data-item-id="lib-square"] button[aria-hidden="true"]').click();
		await expect(dialog.getByTestId('picker-selected-count')).toHaveText('3 selected');
		await dialog.locator('[data-item-id="lib-square"] button[aria-hidden="true"]').click();
		await expect(dialog.getByTestId('picker-selected-count')).toHaveText('2 selected');
		await expect(dialog.getByTestId('picker-preview')).toHaveCount(0);

		await dialog.getByRole('button', { name: 'Preview teapot.png' }).click();
		await expect(dialog.getByTestId('picker-preview')).toBeVisible();
		await expect(dialog.getByTestId('picker-selected-count')).toHaveText('2 selected');
		const previewImage = dialog.getByTestId('picker-preview').locator('img');
		await expect(previewImage).toBeVisible();
		await expect.poll(() => previewImage.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);

		await screenshot(page, JOURNEY, `multi-${viewport.name}`);

		await dialog.locator('[data-item-id="lib-audio"]').scrollIntoViewIfNeeded();
		await page.waitForTimeout(400);
		await expect(dialog.locator('[data-item-id="lib-audio"]').getByText('Only image and video files can be selected here')).toBeVisible();
		await screenshot(page, JOURNEY, `multi-audio-${viewport.name}`);

		await dialog.getByRole('button', { name: 'Use selected' }).click();
		const confirmed = await page.evaluate(() => (window as any).__pickerEvents.confirmed);
		expect(confirmed).toEqual(['lib-portrait', 'lib-landscape']);
	});
}
