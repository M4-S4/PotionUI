import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';

const mockApi = vi.hoisted(() => ({
	listLibraryItems: vi.fn(),
	getLibraryFacets: vi.fn(),
	getTags: vi.fn(),
	setLibraryFavorite: vi.fn(),
	listCollections: vi.fn()
}));

vi.mock('$lib/services/api/index', () => ({ api: mockApi }));

import { libraryStore } from './library';
import { libraryCollectionsStore } from './collections';

function row(id: string, isFavorite = false) {
	return {
		id,
		filename: `${id}.png`,
		original_filename: `${id}.png`,
		media_type: 'image',
		url: `/api/media/uploads/${id}.png`,
		tags: [],
		is_favorite: isFavorite
	};
}

function serveCollections(favorites: number) {
	mockApi.listCollections.mockResolvedValue({
		success: true,
		data: { collections: [], smart_counts: { all: 3, favorites, unsorted: 0 } }
	});
}

async function loadPage(items: ReturnType<typeof row>[]) {
	mockApi.listLibraryItems.mockResolvedValue({
		success: true,
		data: { items, total: items.length, limit: 24, offset: 0 }
	});
	await libraryStore.load();
}

describe('library favorites', () => {
	beforeEach(async () => {
		vi.clearAllMocks();
		libraryStore.reset();
		libraryCollectionsStore.reset();
		mockApi.getLibraryFacets.mockResolvedValue({ success: false });
		mockApi.setLibraryFavorite.mockResolvedValue({ success: true, data: { id: 'a', is_favorite: true } });
		serveCollections(0);
		await loadPage([row('a'), row('b', true), row('c')]);
	});

	it('sends the new flag to the library route', async () => {
		await libraryStore.setItemFavorite('a', true);

		expect(mockApi.setLibraryFavorite).toHaveBeenCalledWith('a', true);
	});

	it('flips the row on the loaded page before the server answers', async () => {
		let release!: (value: unknown) => void;
		mockApi.setLibraryFavorite.mockReturnValue(new Promise((resolve) => (release = resolve)));

		const pending = libraryStore.setItemFavorite('a', true);

		expect(get(libraryStore).items.find((item) => item.id === 'a')?.is_favorite).toBe(true);
		release({ success: true });
		await pending;
	});

	it('patches the open details item too', async () => {
		libraryStore.setSelectedItem(get(libraryStore).items[0]);

		await libraryStore.setItemFavorite('a', true);

		expect(get(libraryStore).selectedItem?.is_favorite).toBe(true);
	});

	it('leaves the other rows alone', async () => {
		await libraryStore.setItemFavorite('a', true);

		const byId = Object.fromEntries(get(libraryStore).items.map((item) => [item.id, item.is_favorite]));
		expect(byId).toEqual({ a: true, b: true, c: false });
	});

	it('restores the previous value and reports failure when the request is refused', async () => {
		mockApi.setLibraryFavorite.mockResolvedValue({ success: false });
		libraryStore.setSelectedItem(get(libraryStore).items[1]);

		const ok = await libraryStore.setItemFavorite('b', false);

		expect(ok).toBe(false);
		expect(get(libraryStore).items.find((item) => item.id === 'b')?.is_favorite).toBe(true);
		expect(get(libraryStore).selectedItem?.is_favorite).toBe(true);
	});

	it('restores the previous value when the request throws', async () => {
		mockApi.setLibraryFavorite.mockRejectedValue(new Error('offline'));

		const ok = await libraryStore.setItemFavorite('a', true);

		expect(ok).toBe(false);
		expect(get(libraryStore).items.find((item) => item.id === 'a')?.is_favorite).toBe(false);
	});

	it('refreshes the sidebar counts so the Favorites view follows', async () => {
		await libraryCollectionsStore.load();
		expect(get(libraryCollectionsStore).smartCounts?.favorites).toBe(0);
		serveCollections(2);

		await libraryStore.setItemFavorite('a', true);

		expect(get(libraryCollectionsStore).smartCounts?.favorites).toBe(2);
	});

	it('does not refresh the counts when the change did not land', async () => {
		mockApi.setLibraryFavorite.mockResolvedValue({ success: false });
		mockApi.listCollections.mockClear();

		await libraryStore.setItemFavorite('a', true);

		expect(mockApi.listCollections).not.toHaveBeenCalled();
	});

	it('drops an un-favorited row from the Favorites view by reloading it', async () => {
		libraryStore.setFilter('favoritesOnly', true);
		await loadPage([row('b', true)]);
		mockApi.listLibraryItems.mockClear();
		mockApi.listLibraryItems.mockResolvedValue({ success: true, data: { items: [], total: 0 } });

		await libraryStore.setItemFavorite('b', false);

		expect(mockApi.listLibraryItems).toHaveBeenCalledWith(
			expect.objectContaining({ favorites_only: true })
		);
		expect(get(libraryStore).items).toEqual([]);
	});

	it('does not reload an ordinary view when something is favorited', async () => {
		mockApi.listLibraryItems.mockClear();

		await libraryStore.setItemFavorite('a', true);

		expect(mockApi.listLibraryItems).not.toHaveBeenCalled();
	});
});
