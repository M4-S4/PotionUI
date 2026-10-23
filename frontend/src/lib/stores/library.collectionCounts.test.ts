import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';

const mockApi = vi.hoisted(() => ({
	listLibraryItems: vi.fn(),
	getLibraryFacets: vi.fn(),
	getTags: vi.fn(),
	deleteLibraryItem: vi.fn(),
	bulkDeleteLibraryItemsByCriteria: vi.fn(),
	listCollections: vi.fn()
}));

vi.mock('$lib/services/api/index', () => ({ api: mockApi }));

import { libraryStore } from './library';
import { libraryCollectionsStore } from './collections';

function serveCollections(count: number) {
	mockApi.listCollections.mockResolvedValue({
		success: true,
		data: {
			collections: [{ id: 'c1', name: 'One', item_count: count }],
			smart_counts: { all: count, favorites: 0, unsorted: 0 }
		}
	});
}

describe('library deletes refresh collection counts', () => {
	beforeEach(async () => {
		vi.clearAllMocks();
		libraryStore.reset();
		libraryCollectionsStore.reset();
		mockApi.listLibraryItems.mockResolvedValue({ success: true, data: { items: [], total: 0 } });
		mockApi.deleteLibraryItem.mockResolvedValue({ success: true });
		mockApi.bulkDeleteLibraryItemsByCriteria.mockResolvedValue({ success: true, data: { deleted_count: 1 } });
		serveCollections(12);
		await libraryCollectionsStore.load();
		serveCollections(11);
	});

	it('single delete', async () => {
		await libraryStore.deleteItem('u1');
		expect(get(libraryCollectionsStore).collections[0].item_count).toBe(11);
	});

	it('bulk delete', async () => {
		libraryStore.toggleSelect('u1');
		await libraryStore.bulkDelete();
		expect(get(libraryCollectionsStore).collections[0].item_count).toBe(11);
	});

	it('delete by criteria', async () => {
		await libraryStore.bulkDeleteByCriteria({} as never);
		expect(get(libraryCollectionsStore).collections[0].item_count).toBe(11);
	});

	describe('when nothing was deleted', () => {
		beforeEach(() => {
			mockApi.listCollections.mockClear();
		});

		it('single delete', async () => {
			mockApi.deleteLibraryItem.mockResolvedValue({ success: false });
			await libraryStore.deleteItem('u1');
			expect(mockApi.listCollections).not.toHaveBeenCalled();
			expect(get(libraryCollectionsStore).collections[0].item_count).toBe(12);
		});

		it('bulk delete', async () => {
			mockApi.deleteLibraryItem.mockResolvedValue({ success: false });
			libraryStore.toggleSelect('u1');
			await libraryStore.bulkDelete();
			expect(mockApi.listCollections).not.toHaveBeenCalled();
			expect(get(libraryCollectionsStore).collections[0].item_count).toBe(12);
		});

		it('delete by criteria', async () => {
			mockApi.bulkDeleteLibraryItemsByCriteria.mockResolvedValue({ success: false });
			await libraryStore.bulkDeleteByCriteria({} as never);
			expect(mockApi.listCollections).not.toHaveBeenCalled();
			expect(get(libraryCollectionsStore).collections[0].item_count).toBe(12);
		});
	});
});
