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

describe('library type counts follow the active view', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		libraryStore.reset();
		mockApi.listLibraryItems.mockResolvedValue({ success: true, data: { items: [], total: 0, limit: 24, offset: 0 } });
		mockApi.getLibraryFacets.mockImplementation(async (favoritesOnly = false) => ({
			success: true,
			data: { media_types: favoritesOnly ? { image: 1 } : { image: 8, video: 4, audio: 1 } }
		}));
	});

	it('counts favourites only after switching to Favorites, and everything after leaving', async () => {
		await libraryStore.loadFacets();
		expect(get(libraryStore).mediaTypeCounts).toEqual({ image: 8, video: 4, audio: 1 });

		libraryStore.setFilter('favoritesOnly', true);
		await libraryStore.load();

		expect(mockApi.getLibraryFacets).toHaveBeenLastCalledWith(true);
		expect(get(libraryStore).mediaTypeCounts).toEqual({ image: 1 });

		libraryStore.setFilter('favoritesOnly', false);
		await libraryStore.load();

		expect(mockApi.getLibraryFacets).toHaveBeenLastCalledWith(false);
		expect(get(libraryStore).mediaTypeCounts).toEqual({ image: 8, video: 4, audio: 1 });
	});

	it('does not refetch counts when the view did not change', async () => {
		await libraryStore.loadFacets();
		mockApi.getLibraryFacets.mockClear();

		await libraryStore.load();

		expect(mockApi.getLibraryFacets).not.toHaveBeenCalled();
	});
});
