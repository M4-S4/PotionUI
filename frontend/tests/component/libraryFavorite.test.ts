// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
vi.mock('$lib/components/plugins/PluginSlot.svelte', async () => ({
	default: (await import('./fixtures/PluginSlotStub.svelte')).default
}));

class StubResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}
vi.stubGlobal('ResizeObserver', StubResizeObserver);

const setLibraryFavorite = vi.fn();

vi.mock('$lib/services/api/index', () => ({
	api: {
		getOrganizeProvenance: vi.fn(() => Promise.resolve({ success: true, data: [] })),
		listLibraryItems: vi.fn().mockResolvedValue({ success: false }),
		getLibraryFacets: vi.fn().mockResolvedValue({ success: false }),
		getTags: vi.fn().mockResolvedValue({ success: false }),
		listCollections: vi.fn().mockResolvedValue({ success: false }),
		setLibraryItemTags: vi.fn(),
		setLibraryFavorite: (...args: unknown[]) => setLibraryFavorite(...args)
	}
}));

const { libraryStore } = await import('$lib/stores/library');
const { default: LibraryItemModal } = await import(
	'../../src/lib/components/library/LibraryItemModal.svelte'
);
const { default: LibraryCard } = await import(
	'../../src/routes/library/components/LibraryCard.svelte'
);
const { createLibraryEntryMenu } = await import('$lib/tools/entryMenu');
const { registerEntryTools } = await import('$lib/tools/entryTools');
const { createClassComponent } = await import('svelte/legacy');
const { flushSync, tick } = await import('svelte');

vi.stubGlobal('ClipboardItem', class {});
registerEntryTools();

async function settle() {
	for (let i = 0; i < 6; i += 1) {
		await tick();
		await Promise.resolve();
	}
}

function item(overrides: Record<string, unknown> = {}) {
	return {
		id: 'row-1',
		filename: 'stored-uuid.png',
		original_filename: 'portrait.png',
		media_type: 'image',
		url: '/api/media/uploads/stored-uuid.png',
		width: 1024,
		height: 1536,
		tags: [],
		...overrides
	};
}

function mountModal(row: Record<string, unknown>) {
	libraryStore.setSelectedItem(row as never);
	const target = document.createElement('div');
	document.body.appendChild(target);
	createClassComponent({
		component: LibraryItemModal as never,
		target,
		props: { item: row, onClose: vi.fn(), onDeleteRequest: vi.fn() }
	});
}

function favoriteButton(): HTMLButtonElement | null {
	return document.body.querySelector<HTMLButtonElement>('button[data-library-favorite]');
}

beforeEach(() => {
	document.body.innerHTML = '';
	libraryStore.reset();
	setLibraryFavorite.mockReset();
	setLibraryFavorite.mockResolvedValue({ success: true, data: { id: 'row-1', is_favorite: true } });
});

describe('Library details favorite', () => {
	it('sits in the details action bar, labelled, and is off for an item that is not a favorite', async () => {
		mountModal(item());
		await settle();

		const button = favoriteButton();
		expect(button).toBeTruthy();
		expect(button!.closest('[data-details-action-bar]')).toBeTruthy();
		expect(button!.textContent).toContain('Favorite');
		expect(button!.getAttribute('aria-pressed')).toBe('false');
	});

	it('shows an item that already is one as on', async () => {
		mountModal(item({ is_favorite: true }));
		await settle();

		expect(favoriteButton()!.getAttribute('aria-pressed')).toBe('true');
		expect(favoriteButton()!.className).toContain('text-signal');
	});

	it('writes the favorite for the row it is showing', async () => {
		mountModal(item());
		await settle();

		favoriteButton()!.click();
		await settle();

		expect(setLibraryFavorite).toHaveBeenCalledTimes(1);
		expect(setLibraryFavorite).toHaveBeenCalledWith('row-1', true);
	});

	it('writes the removal for a favorite', async () => {
		mountModal(item({ is_favorite: true }));
		await settle();

		favoriteButton()!.click();
		await settle();

		expect(setLibraryFavorite).toHaveBeenCalledWith('row-1', false);
	});
});

describe('Library card menu favorite', () => {
	const host = { open: vi.fn(), remove: vi.fn(), deselect: vi.fn(), addToCollection: vi.fn(), favorite: vi.fn() };
	let current: Record<string, unknown>;
	const entryMenu = createLibraryEntryMenu({
		items: () => [current as never],
		selectedIds: () => [],
		host,
		onTool: vi.fn()
	});
	let component: { $destroy: () => void } | undefined;

	function openMenu() {
		const target = document.createElement('div');
		document.body.appendChild(target);
		component = createClassComponent({
			component: LibraryCard as never,
			target,
			props: { item: current, entryMenu, tile: { width: 320, height: 160 } }
		});
		target.querySelector<HTMLButtonElement>('[data-entry-menu-trigger]')!.click();
		flushSync();
	}

	function favoriteRow(): HTMLButtonElement | null {
		return document.body.querySelector<HTMLButtonElement>('[data-tool="favorite"]');
	}

	afterEach(() => {
		component?.$destroy();
		component = undefined;
		host.favorite.mockReset();
	});

	it('shows Favorite in the Organize group, ahead of Add to collection', async () => {
		current = item();
		openMenu();
		await settle();

		const ids = Array.from(document.body.querySelectorAll('[data-tool]')).map((el) =>
			el.getAttribute('data-tool')
		);
		expect(ids.indexOf('favorite')).toBeGreaterThan(-1);
		expect(ids.indexOf('favorite')).toBeLessThan(ids.indexOf('add-to-collection'));
		expect(favoriteRow()!.textContent).toContain('Favorite');
		expect(favoriteRow()!.textContent).not.toContain('Remove');
	});

	it('reads Remove favorite for a favorite', async () => {
		current = item({ is_favorite: true });
		openMenu();
		await settle();

		expect(favoriteRow()!.textContent).toContain('Remove favorite');
	});

	it('hands the item to the page host when picked', async () => {
		current = item();
		openMenu();
		await settle();

		favoriteRow()!.click();
		await settle();

		expect(host.favorite).toHaveBeenCalledTimes(1);
		expect(host.favorite.mock.calls[0][0].items[0].id).toBe('row-1');
	});
});
