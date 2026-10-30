// @vitest-environment jsdom
//
// The library card shares the generation card's width-bucketed chrome
// (generationCardChrome.ts) but shows an upload's own affordances. The wiring
// worth pinning down is what a given tile width actually renders - the pure
// bucket table can't tell you whether the card asks it at all - and that the
// name shown is never the on-disk uuid.
import { describe, it, expect, vi, afterEach } from 'vitest';

const { default: LibraryCard } = await import(
	'../../src/routes/library/components/LibraryCard.svelte'
);
const { createClassComponent } = await import('svelte/legacy');
const { registerEntryTools } = await import('$lib/tools/entryTools');
const { registerMediaTool } = await import('$lib/tools/tools');
const { createLibraryEntryMenu } = await import('$lib/tools/entryMenu');
const { flushSync } = await import('svelte');
const { libraryStore } = await import('$lib/stores/library');

vi.stubGlobal('ClipboardItem', class {});
registerEntryTools();
registerMediaTool({
	id: 'edit-image',
	label: 'Edit image',
	icon: 'paint-brush',
	category: 'compose',
	source: 'core',
	scopes: ['history', 'library'],
	selection: { min: 1, max: 1 },
	kinds: ['image'],
	applies: () => ({ enabled: true }),
	component: {} as never
});

const host = {
	open: vi.fn(),
	remove: vi.fn(),
	deselect: vi.fn(),
	favorite: vi.fn(),
	addToCollection: vi.fn()
};
const entryMenu = createLibraryEntryMenu({
	items: () => [ITEM as never],
	selectedIds: () => [],
	collections: () => [{ id: 'c1', name: 'Boats' }],
	host,
	onTool: vi.fn()
});

const ITEM = {
	id: 'item-1',
	filename: '0d3f8e2a-1111-2222-3333-444455556666.png',
	original_filename: 'sunset.png',
	media_type: 'image',
	url: '/api/media/uploads/0d3f8e2a-1111-2222-3333-444455556666.png',
	width: 1024,
	height: 512,
	size: 2048,
	created_at: '2026-08-13T10:00:00Z',
	tags: []
};

function mountCard(props: Record<string, unknown> = {}) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: LibraryCard as never,
		target,
		props: { item: ITEM, entryMenu, tile: { width: 320, height: 160 }, ...props }
	});
	return {
		target,
		component,
		labels: () =>
			Array.from(target.querySelectorAll('button')).map((b) => b.getAttribute('aria-label') ?? ''),
		text: () => target.textContent ?? '',
		destroy: () => {
			component.$destroy();
			target.remove();
		}
	};
}

let mounted: ReturnType<typeof mountCard> | undefined;

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
});

describe('library card', () => {
	it('shows the original filename, never the on-disk uuid', () => {
		mounted = mountCard();

		expect(mounted.text()).toContain('sunset.png');
		expect(mounted.text()).not.toContain('0d3f8e2a');
	});

	it('renders the item resolution', () => {
		mounted = mountCard();

		expect(mounted.text()).toContain('1024×512');
	});

	it('hover row is exactly the Favorite toggle and the menu button on a wide tile', () => {
		mounted = mountCard({ tile: { width: 320, height: 160 } });

		expect(mounted.labels()).toEqual(['Select item', 'Favorite', 'Item actions']);
	});

	it('retires the grouped pill toolbar: no view or download buttons on the card', () => {
		mounted = mountCard({ tile: { width: 320, height: 160 } });

		const labels = mounted.labels();
		expect(labels).not.toContain('Open library item');
		expect(labels).not.toContain('Download');
		expect(mounted.target.querySelector('.ring-inset')).toBeNull();
	});

	it('keeps the controls out of the media element', () => {
		mounted = mountCard();
		const media = mounted.target.querySelector('.media-zoom')!;
		expect(media.querySelectorAll('button, [role="button"]').length).toBe(0);
		expect(media.contains(mounted.target.querySelector('[data-entry-menu-trigger]'))).toBe(false);
	});

	it('shows a Favorite toggle on the tile, signal blue and pressed for a favorite', () => {
		mounted = mountCard({ item: { ...ITEM, is_favorite: true } });

		const toggle = mounted.target.querySelector<HTMLButtonElement>('[data-entry-favorite]')!;
		expect(toggle.getAttribute('aria-label')).toBe('Remove favorite');
		expect(toggle.getAttribute('aria-pressed')).toBe('true');
		expect(toggle.className).toContain('text-signal');
	});

	it('leaves the Favorite toggle off for an item that is not a favorite', () => {
		mounted = mountCard();

		const toggle = mounted.target.querySelector<HTMLButtonElement>('[data-entry-favorite]')!;
		expect(toggle.getAttribute('aria-pressed')).toBe('false');
	});

	it('keeps the Favorite toggle and the menu button on narrow tiles', () => {
		mounted = mountCard({ tile: { width: 120, height: 120 } });
		expect(mounted.labels()).toContain('Favorite');
		expect(mounted.labels()).toContain('Item actions');
		mounted.destroy();
		mounted = mountCard({ tile: { width: 100, height: 100 } });
		expect(mounted.labels()).toContain('Favorite');
		expect(mounted.labels()).toContain('Item actions');
	});

	it('opens the same grouped menu as History, minus what an upload cannot do', async () => {
		mounted = mountCard();
		(mounted.target.querySelector('[data-entry-menu-trigger]') as HTMLButtonElement).click();
		for (let i = 0; i < 4; i++) await new Promise((resolve) => setTimeout(resolve, 0));
		flushSync();

		const ids = Array.from(document.body.querySelectorAll('[data-tool]')).map((el) => el.getAttribute('data-tool'));
		expect(ids).toEqual(['open-details', 'edit-image', 'download', 'copy-image', 'favorite', 'add-to-collection', 'delete']);
		document.body.querySelector<HTMLButtonElement>('[data-tool="open-details"]')!.click();
		expect(host.open).toHaveBeenCalledTimes(1);
	});

	it('hands Delete to the host from the menu rather than deleting anything itself', async () => {
		mounted = mountCard();
		(mounted.target.querySelector('[data-entry-menu-trigger]') as HTMLButtonElement).click();
		for (let i = 0; i < 4; i++) await new Promise((resolve) => setTimeout(resolve, 0));
		flushSync();

		document.body.querySelector<HTMLButtonElement>('[data-tool="delete"]')!.click();

		expect(host.remove).toHaveBeenCalledTimes(1);
	});

	it('writes the favorite through the store when the toggle is pressed', async () => {
		const spy = vi.spyOn(libraryStore, 'setItemFavorite').mockResolvedValue(true);
		mounted = mountCard();

		mounted.target.querySelector<HTMLButtonElement>('[data-entry-favorite]')!.click();

		expect(spy).toHaveBeenCalledWith('item-1', true);
		spy.mockRestore();
	});

	it('leaves only the menu button in selection mode', () => {
		mounted = mountCard({ selectable: true });

		expect(mounted.labels()).not.toContain('Favorite');
		expect(mounted.labels()).toContain('Item actions');
	});
});
