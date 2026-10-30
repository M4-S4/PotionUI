import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';

const mockApi = vi.hoisted(() => ({
	listLibraryItems: vi.fn(),
	getLibraryFacets: vi.fn(),
	getTags: vi.fn(),
	listCollections: vi.fn()
}));

vi.mock('$lib/services/api/index', () => ({ api: mockApi }));

import { libraryStore } from './library';

const ALL = { image: 8, video: 4, audio: 1 };

describe('library type counts follow the active view', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		libraryStore.reset();
		mockApi.listLibraryItems.mockResolvedValue({ success: true, data: { items: [], total: 0, limit: 24, offset: 0 } });
		mockApi.getLibraryFacets.mockImplementation(async (scope: Record<string, unknown> = {}) => {
			if (scope.favorites_only) return { success: true, data: { media_types: { image: 1 } } };
			if (scope.collection_id) return { success: true, data: { media_types: { video: 2 } } };
			if (scope.unsorted) return { success: true, data: { media_types: { audio: 1 } } };
			return { success: true, data: { media_types: ALL } };
		});
	});

	async function switchTo(key: 'favoritesOnly' | 'collectionId' | 'unsorted' | 'directOnly', value: unknown) {
		libraryStore.setFilter(key, value as never);
		await libraryStore.load();
	}

	it('counts favourites only after switching to Favorites, and everything after leaving', async () => {
		await libraryStore.loadFacets();
		expect(get(libraryStore).mediaTypeCounts).toEqual(ALL);

		await switchTo('favoritesOnly', true);
		expect(mockApi.getLibraryFacets).toHaveBeenLastCalledWith({ favorites_only: true });
		expect(get(libraryStore).mediaTypeCounts).toEqual({ image: 1 });

		await switchTo('favoritesOnly', false);
		expect(mockApi.getLibraryFacets).toHaveBeenLastCalledWith({});
		expect(get(libraryStore).mediaTypeCounts).toEqual(ALL);
	});

	it('counts a collection view and the direct-only variant', async () => {
		await switchTo('collectionId', 'c1');
		expect(mockApi.getLibraryFacets).toHaveBeenLastCalledWith({ collection_id: 'c1' });
		expect(get(libraryStore).mediaTypeCounts).toEqual({ video: 2 });

		await switchTo('directOnly', true);
		expect(mockApi.getLibraryFacets).toHaveBeenLastCalledWith({
			collection_id: 'c1',
			include_descendants: false
		});
	});

	it('counts the Unsorted view', async () => {
		await switchTo('unsorted', true);

		expect(mockApi.getLibraryFacets).toHaveBeenLastCalledWith({ unsorted: true });
		expect(get(libraryStore).mediaTypeCounts).toEqual({ audio: 1 });
	});

	it('does not refetch counts when the view did not change', async () => {
		await libraryStore.loadFacets();
		mockApi.getLibraryFacets.mockClear();

		await libraryStore.load();

		expect(mockApi.getLibraryFacets).not.toHaveBeenCalled();
	});
});
