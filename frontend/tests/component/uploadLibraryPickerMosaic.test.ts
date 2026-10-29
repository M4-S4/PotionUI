// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';

const apiMocks = vi.hoisted(() => ({
	listLibraryItems: vi.fn(),
	deleteLibraryItem: vi.fn(),
	getTags: vi.fn(),
	getClient: () => ({ get: vi.fn().mockResolvedValue({ data: { data: {} } }) })
}));

vi.mock('$lib/services/api/index', () => ({
	api: new Proxy(apiMocks as Record<string, any>, {
		get: (target, key: string) =>
			key in target ? target[key] : vi.fn().mockResolvedValue({ success: false })
	})
}));

const { default: UploadLibraryModal } = await import('$lib/components/modals/UploadLibraryModal.svelte');
const { createClassComponent } = await import('svelte/legacy');
const { tick } = await import('svelte');
const { stubClientWidth } = await import('./helpers/stubClientWidth');
const { formatSeconds } = await import('$lib/utils/format');

function libraryItem(id: string, mediaType: string, name: string, extra: Record<string, unknown> = {}) {
	return {
		id,
		filename: `stored-${name}`,
		original_filename: name,
		media_type: mediaType,
		url: `/media/${id}`,
		tags: [],
		...extra
	};
}

const ITEMS = [
	libraryItem('wide', 'image', 'wide.png', { width: 1600, height: 900, thumbnail_medium: '/thumb/wide.png' }),
	libraryItem('tall', 'image', 'tall.png', { width: 900, height: 1600, thumbnail_medium: '/thumb/tall.png' }),
	libraryItem('square', 'image', 'square.png', { width: 1000, height: 1000 }),
	libraryItem('clip', 'video', 'clip.mp4', {
		width: 1280,
		height: 720,
		duration_seconds: 8,
		thumbnail_medium: '/thumb/clip.png'
	}),
	libraryItem('bare', 'video', 'bare.mp4'),
	libraryItem('song', 'audio', 'song.wav', { duration_seconds: 12 })
];

const FIRST_ROW = ['wide', 'tall', 'square', 'clip'];

let components: Array<{ $destroy: () => void }> = [];
let restoreWidth: () => void = () => {};

async function mount(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	components.push(
		createClassComponent({
			component: UploadLibraryModal as never,
			target,
			props: { isOpen: true, onClose: () => {}, ...props }
		})
	);
	await vi.waitFor(() =>
		expect(document.querySelectorAll('[data-testid="library-file-tile"]').length).toBeGreaterThan(0)
	);
	await tick();
}

function tile(id: string): HTMLElement {
	const el = document.querySelector<HTMLElement>(`[data-item-id="${id}"]`);
	if (!el) throw new Error(`no tile for ${id}`);
	return el;
}

function frame(id: string): HTMLElement {
	return tile(id).querySelector<HTMLElement>('.media-zoom')!;
}

function tileWidth(id: string): number {
	return parseFloat(tile(id).style.width);
}

function frameHeight(id: string): number {
	return parseFloat(frame(id).style.height);
}

function checkbox(name: string): HTMLInputElement {
	const el = document.querySelector<HTMLInputElement>(`input[aria-label="Select ${name}"]`);
	if (!el) throw new Error(`no checkbox for ${name}`);
	return el;
}

function tileButton(id: string): HTMLButtonElement {
	return tile(id).querySelector<HTMLButtonElement>('button[aria-hidden="true"]')!;
}

function count(): string {
	return document.querySelector('[data-testid="picker-selected-count"]')?.textContent?.trim() ?? '';
}

function buttonByText(text: string): HTMLButtonElement | undefined {
	return [...document.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === text);
}

beforeEach(() => {
	restoreWidth = stubClientWidth(1000);
	apiMocks.listLibraryItems.mockResolvedValue({ success: true, data: { items: ITEMS, total: ITEMS.length } });
	apiMocks.deleteLibraryItem.mockResolvedValue({ success: true, data: { id: 'square', deleted: true } });
	apiMocks.getTags.mockResolvedValue({ success: true, data: { tags: [] } });
});

afterEach(() => {
	restoreWidth();
	components.forEach((c) => c.$destroy());
	components = [];
	document.body.innerHTML = '';
	vi.clearAllMocks();
});

describe('library picker mosaic', () => {
	it.each([false, true])('renders Library cards at native aspect ratios (multiple=%s)', async (multiple) => {
		await mount({ multiple });

		expect(document.querySelectorAll('[data-library-card-body]').length).toBe(ITEMS.length);
		expect(new Set(FIRST_ROW.map(frameHeight)).size).toBe(1);
		expect(tileWidth('wide') / frameHeight('wide')).toBeCloseTo(16 / 9, 1);
		expect(tileWidth('tall') / frameHeight('tall')).toBeCloseTo(9 / 16, 1);
		expect(tileWidth('square') / frameHeight('square')).toBeCloseTo(1, 1);
		expect(tileWidth('clip') / frameHeight('clip')).toBeCloseTo(16 / 9, 1);
		expect(tileWidth('tall')).toBeLessThan(tileWidth('wide'));
	});

	it('justifies each row edge to edge and keeps the last row at the preferred height', async () => {
		await mount({ multiple: true });

		const rowWidth = FIRST_ROW.reduce((sum, id) => sum + tileWidth(id), 0) + 12 * (FIRST_ROW.length - 1);
		expect(rowWidth).toBeGreaterThan(1000 - FIRST_ROW.length);
		expect(rowWidth).toBeLessThanOrEqual(1000);
		expect(frameHeight('wide')).toBeLessThan(217);

		expect(frameHeight('bare')).toBe(217);
		expect(frameHeight('song')).toBe(217);
		const wrappers = [...document.querySelectorAll<HTMLElement>('[data-testid="library-mosaic"] > div')];
		expect(wrappers.map((el) => el.getAttribute('data-row-end'))).toEqual([
			'false',
			'false',
			'false',
			'true',
			'false',
			'true'
		]);
		expect(wrappers.map((el) => el.style.flex.split(' ').slice(0, 2).join(' '))).toEqual([
			'0 0',
			'0 0',
			'0 0',
			'1 1',
			'0 0',
			'1 1'
		]);
	});

	it('never crops an image: the frame has the native aspect and the picture is contained', async () => {
		await mount({ multiple: true });

		for (const id of ['wide', 'tall', 'square']) {
			const img = frame(id).querySelector<HTMLImageElement>('img')!;
			expect(img.classList.contains('object-contain')).toBe(true);
			expect(img.classList.contains('object-cover')).toBe(false);
		}
		expect(frame('wide').querySelector('img')!.getAttribute('src')).toBe('/thumb/wide.png');
		expect(frame('square').querySelector('img')!.getAttribute('src')).toBe('/media/square');
	});

	it('shows a video through its thumbnail, a bare video as a square, and audio as a placeholder', async () => {
		await mount({ multiple: true });

		expect(frame('clip').querySelector('img')!.getAttribute('src')).toBe('/thumb/clip.png');
		expect(frame('clip').querySelector('video')).toBeNull();

		expect(frame('bare').querySelector('video')).not.toBeNull();
		expect(tileWidth('bare')).toBeCloseTo(frameHeight('bare'), 0);

		expect(frame('song').querySelector('img, video')).toBeNull();
		expect(tileWidth('song')).toBeCloseTo(frameHeight('song'), 0);
		expect(frame('song').textContent).toContain(formatSeconds(12));
	});

	it('captions each tile with the file name', async () => {
		await mount({ multiple: true });
		expect(tile('wide').textContent).toContain('wide.png');
		expect(tile('song').textContent).toContain('song.wav');
	});

	it('renders each card inert so only the tile controls take input', async () => {
		await mount({ multiple: false });
		for (const id of ['wide', 'clip', 'song']) {
			const wrapper = tile(id).querySelector('span[inert]')!;
			expect(wrapper.querySelector('[data-library-card-body]')).not.toBeNull();
		}
	});

	it('shows the empty state instead of a mosaic when nothing matches', async () => {
		apiMocks.listLibraryItems.mockResolvedValue({ success: true, data: { items: [], total: 0 } });
		const target = document.createElement('div');
		document.body.appendChild(target);
		components.push(
			createClassComponent({
				component: UploadLibraryModal as never,
				target,
				props: { isOpen: true, onClose: () => {}, multiple: true }
			})
		);
		await vi.waitFor(() => expect(document.body.textContent).toContain('Nothing here yet'));
		expect(document.querySelector('[data-testid="library-mosaic"]')).toBeNull();
		expect(document.querySelectorAll('[data-library-card-body]').length).toBe(0);
	});
});

describe('library picker mosaic multi-select', () => {
	it('toggles selection from the labelled checkbox and counts it', async () => {
		const onSelectionChange = vi.fn();
		await mount({ multiple: true, onSelectionChange });

		const box = checkbox('tall.png');
		box.click();
		await tick();

		expect(onSelectionChange).toHaveBeenLastCalledWith(['tall'], expect.any(Array));
		expect(box.checked).toBe(true);
		expect(box.closest('label')).not.toBeNull();
		expect(tile('tall').getAttribute('data-selected')).toBe('true');
		expect(tile('wide').getAttribute('data-selected')).toBe('false');
		expect(count()).toBe('1 selected');
		expect(checkbox('wide.png').checked).toBe(false);
	});

	it('toggles selection when the tile is clicked and keeps the checkbox in sync', async () => {
		const onSelectionChange = vi.fn();
		await mount({ multiple: true, onSelectionChange });

		tileButton('clip').click();
		await tick();
		expect(onSelectionChange).toHaveBeenLastCalledWith(['clip'], expect.any(Array));
		expect(checkbox('clip.mp4').checked).toBe(true);
		expect(tile('clip').getAttribute('data-selected')).toBe('true');
		expect(document.querySelector('[data-testid="picker-preview"]')).toBeNull();

		tileButton('clip').click();
		await tick();
		expect(onSelectionChange).toHaveBeenLastCalledWith([], expect.any(Array));
		expect(checkbox('clip.mp4').checked).toBe(false);
	});

	it('marks a selected card so the mosaic shows which tiles are picked', async () => {
		await mount({ multiple: true, selectedKeys: ['wide'] });
		expect(tile('wide').querySelector('.bg-signal\\/15')).not.toBeNull();
		expect(tile('tall').querySelector('.bg-signal\\/15')).toBeNull();
	});

	it('opens the preview from the magnifier without selecting', async () => {
		const onSelectionChange = vi.fn();
		await mount({ multiple: true, onSelectionChange });

		tile('wide').querySelector<HTMLButtonElement>('button[aria-label="Preview wide.png"]')!.click();
		await tick();

		const preview = document.querySelector('[data-testid="picker-preview"]');
		expect(preview).not.toBeNull();
		expect(preview!.querySelector('img')!.getAttribute('src')).toBe('/media/wide');
		expect(onSelectionChange).not.toHaveBeenCalled();
		expect(checkbox('wide.png').checked).toBe(false);
	});

	it('disables media outside the type constraint and states the reason on that tile', async () => {
		const onSelectionChange = vi.fn();
		await mount({ multiple: true, onSelectionChange, selectableMediaTypes: ['image'] });

		const box = checkbox('song.wav');
		expect(box.disabled).toBe(true);
		expect(tile('song').textContent).toContain('Only image files can be selected here');
		expect(tile('song').querySelector('.opacity-50')).not.toBeNull();
		const reason = document.getElementById(box.getAttribute('aria-describedby')!);
		expect(reason?.textContent).toContain('Only image files can be selected here');

		box.click();
		expect(tileButton('song').disabled).toBe(true);
		tileButton('song').click();
		expect(onSelectionChange).not.toHaveBeenCalled();

		expect(checkbox('wide.png').disabled).toBe(false);
		expect(tile('wide').textContent).not.toContain('Only image files can be selected here');
		expect(tile('wide').querySelector('.opacity-50')).toBeNull();
	});

	it('keeps the preview available on a disabled tile', async () => {
		await mount({ multiple: true, selectableMediaTypes: ['image'] });
		tile('song').querySelector<HTMLButtonElement>('button[aria-label="Preview song.wav"]')!.click();
		await tick();
		expect(document.querySelector('[data-testid="picker-preview"]')).not.toBeNull();
	});

	it('confirms the picked tiles', async () => {
		const onConfirm = vi.fn();
		const onClose = vi.fn();
		await mount({ multiple: true, onConfirm, onClose });

		expect(buttonByText('Use selected')!.disabled).toBe(true);
		checkbox('tall.png').click();
		checkbox('song.wav').click();
		await tick();
		buttonByText('Use selected')!.click();

		expect(onConfirm).toHaveBeenCalledWith(['tall', 'song'], expect.any(Array));
		expect(onClose).toHaveBeenCalled();
	});
});

describe('library picker mosaic single-pick', () => {
	it('offers button tiles without checkboxes, magnifiers or a selection bar', async () => {
		await mount({ multiple: false });

		expect(document.querySelector('input[type="checkbox"]')).toBeNull();
		expect(document.querySelector('[data-testid="picker-selected-count"]')).toBeNull();
		expect(document.querySelector('button[aria-label^="Preview "]')).toBeNull();
		expect(document.querySelectorAll('button[aria-label^="Use "]').length).toBe(ITEMS.length);
	});

	it('hands the clicked library item to onSelect', async () => {
		const onSelect = vi.fn();
		await mount({ multiple: false, onSelect });

		document.querySelector<HTMLButtonElement>('button[aria-label="Use tall.png"]')!.click();
		expect(onSelect).toHaveBeenCalledTimes(1);
		expect(onSelect.mock.calls[0][0]).toMatchObject({ id: 'tall', original_filename: 'tall.png' });

		document.querySelector<HTMLButtonElement>('button[aria-label="Use song.wav"]')!.click();
		expect(onSelect.mock.calls[1][0]).toMatchObject({ id: 'song', media_type: 'audio' });
	});

	it('keeps the inline delete confirmation on a tile', async () => {
		const onSelect = vi.fn();
		await mount({ multiple: false, onSelect });

		document.querySelector<HTMLButtonElement>('button[aria-label="Delete square.png"]')!.click();
		await tick();
		expect(buttonByText('Delete')).toBeDefined();
		expect(onSelect).not.toHaveBeenCalled();

		buttonByText('Cancel')!.click();
		await tick();
		expect(buttonByText('Delete')).toBeUndefined();
		expect(apiMocks.deleteLibraryItem).not.toHaveBeenCalled();
		expect(document.querySelector('button[aria-label="Delete square.png"]')).not.toBeNull();

		document.querySelector<HTMLButtonElement>('button[aria-label="Delete square.png"]')!.click();
		await tick();
		buttonByText('Delete')!.click();
		await vi.waitFor(() => expect(document.querySelector('[data-item-id="square"]')).toBeNull());
		expect(apiMocks.deleteLibraryItem).toHaveBeenCalledWith('square');
		expect(document.querySelectorAll('[data-testid="library-file-tile"]').length).toBe(ITEMS.length - 1);
	});
});
