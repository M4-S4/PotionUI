import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';

const mockApi = vi.hoisted(() => ({
	getGenerationHistory: vi.fn(),
	getHistoryFacets: vi.fn(),
	getTags: vi.fn(),
	deleteGenerationHistory: vi.fn(),
	bulkDeleteGenerations: vi.fn(),
	bulkDeleteByCriteria: vi.fn(),
	listCollections: vi.fn()
}));
const mockDeleteGridById = vi.fn();

vi.mock('$lib/services/api/index', () => ({ api: mockApi }));
vi.mock('$lib/generation/compare/compareApi', () => ({
	deleteGridById: (...args: unknown[]) => mockDeleteGridById(...args)
}));

import { historyStore } from './history';
import { historyCollectionsStore } from './collections';

const solo = { id: 'solo', status: 'completed', files: [] };
const grid = { id: 'grid-1', cols: 2, rows: 1, cell_count: 2 };
const gridRep = { id: 'rep', status: 'completed', files: [], grid_id: 'grid-1', grid };

function serveCollections(count: number) {
	mockApi.listCollections.mockResolvedValue({
		success: true,
		data: {
			collections: [{ id: 'c1', name: 'One', item_count: count }],
			smart_counts: { all: count, favorites: 0, unsorted: 0 }
		}
	});
}

describe('history deletes refresh collection counts', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		historyStore.reset();
		historyCollectionsStore.reset();
		mockApi.getGenerationHistory.mockResolvedValue({
			success: true,
			data: { generations: [solo, gridRep], total: 2 }
		});
		mockApi.deleteGenerationHistory.mockResolvedValue({ success: true });
		mockApi.bulkDeleteGenerations.mockResolvedValue({ success: true });
		mockApi.bulkDeleteByCriteria.mockResolvedValue({ success: true, data: { deleted_count: 1 } });
		mockDeleteGridById.mockResolvedValue(undefined);
		serveCollections(12);
	});

	async function loadedAt(count: number) {
		await historyCollectionsStore.load();
		expect(get(historyCollectionsStore).collections[0].item_count).toBe(count);
		serveCollections(count - 1);
		mockApi.listCollections.mockClear();
	}

	it('single delete', async () => {
		await historyStore.loadGenerations();
		await loadedAt(12);
		await historyStore.deleteGeneration('solo');
		expect(get(historyCollectionsStore).collections[0].item_count).toBe(11);
		expect(get(historyCollectionsStore).smartCounts?.all).toBe(11);
	});

	it('failed single delete leaves counts alone', async () => {
		await historyStore.loadGenerations();
		await loadedAt(12);
		mockApi.deleteGenerationHistory.mockResolvedValue({ success: false });
		await historyStore.deleteGeneration('solo');
		expect(mockApi.listCollections).not.toHaveBeenCalled();
	});

	it('compare grid delete', async () => {
		await historyStore.loadGenerations();
		await loadedAt(12);
		await historyStore.deleteGrid('grid-1');
		expect(get(historyCollectionsStore).collections[0].item_count).toBe(11);
	});

	it('bulk delete', async () => {
		await historyStore.loadGenerations();
		await loadedAt(12);
		historyStore.toggleGenerationSelection('solo');
		await historyStore.bulkDeleteGenerations();
		expect(get(historyCollectionsStore).collections[0].item_count).toBe(11);
	});

	it('delete by criteria', async () => {
		await historyStore.loadGenerations();
		await loadedAt(12);
		await historyStore.bulkDeleteByCriteria({} as never);
		expect(get(historyCollectionsStore).collections[0].item_count).toBe(11);
	});
});
