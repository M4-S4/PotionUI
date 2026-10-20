import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';

const mockGetGenerationHistory = vi.fn();
const mockBulkDeleteGenerations = vi.fn();
const mockDeleteGridById = vi.fn();

vi.mock('$lib/services/api/index', () => ({
	api: {
		getGenerationHistory: (...args: unknown[]) => mockGetGenerationHistory(...args),
		getHistoryFacets: vi.fn(),
		getTags: vi.fn(),
		bulkDeleteGenerations: (...args: unknown[]) => mockBulkDeleteGenerations(...args)
	}
}));

vi.mock('$lib/generation/compare/compareApi', () => ({
	deleteGridById: (...args: unknown[]) => mockDeleteGridById(...args)
}));

import { historyStore } from './history';
import {
	bulkDeleteMessage,
	deleteCopy,
	gridDeleteMessage,
	planBulkDelete
} from '$lib/utils/historyDeleteCopy';
import type { GenerationHistoryItem } from '$lib/types/history';

const gridOne = { id: 'grid-1', cols: 3, rows: 2, cell_count: 6 };
const gridTwo = { id: 'grid-2', cols: 2, rows: 2, cell_count: 4 };
const solo = { id: 'solo', status: 'completed', files: [{}] };
const repOne = { id: 'rep', status: 'completed', files: [], grid_id: 'grid-1', grid: gridOne };
const repTwo = { id: 'rep2', status: 'completed', files: [], grid_id: 'grid-2', grid: gridTwo };
const items = [solo, repOne, repTwo] as unknown as GenerationHistoryItem[];

function serve(list: unknown[]) {
	mockGetGenerationHistory.mockResolvedValue({
		success: true,
		data: { generations: list, total: list.length }
	});
}

describe('X/Y compare deletion', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		historyStore.reset();
		serve(items);
		mockDeleteGridById.mockResolvedValue(undefined);
		mockBulkDeleteGenerations.mockResolvedValue({ success: true });
	});

	it('deleteGrid sends exactly one grid request and drops the card before it settles', async () => {
		await historyStore.loadGenerations();
		let release: () => void = () => {};
		mockDeleteGridById.mockReturnValue(new Promise<void>((resolve) => (release = resolve)));

		const pending = historyStore.deleteGrid('grid-1');
		expect(get(historyStore).generations.map((gen) => gen.id)).toEqual(['solo', 'rep2']);
		serve([solo, repTwo]);
		release();
		await pending;

		expect(mockDeleteGridById).toHaveBeenCalledTimes(1);
		expect(mockDeleteGridById).toHaveBeenCalledWith('grid-1');
		expect(mockBulkDeleteGenerations).not.toHaveBeenCalled();
	});

	it('deleteGrid restores the card in place and the selection when the request fails', async () => {
		await historyStore.loadGenerations();
		historyStore.toggleGenerationSelection('rep');
		mockDeleteGridById.mockRejectedValue(new Error('boom'));

		await expect(historyStore.deleteGrid('grid-1')).rejects.toThrow('boom');

		const state = get(historyStore);
		expect(state.generations.map((gen) => gen.id)).toEqual(['solo', 'rep', 'rep2']);
		expect(state.totalCount).toBe(3);
		expect(state.selectedGenerationIds).toEqual(['rep']);
	});

	it('bulk delete sends each compare as one grid request and only loose ids to the bulk endpoint', async () => {
		await historyStore.loadGenerations();
		historyStore.toggleGenerationSelection('rep');
		historyStore.toggleGenerationSelection('solo');
		serve([repTwo]);

		await historyStore.bulkDeleteGenerations();

		expect(mockDeleteGridById).toHaveBeenCalledTimes(1);
		expect(mockDeleteGridById).toHaveBeenCalledWith('grid-1');
		expect(mockBulkDeleteGenerations).toHaveBeenCalledTimes(1);
		expect(mockBulkDeleteGenerations).toHaveBeenCalledWith(['solo']);
		expect(get(historyStore).selectedGenerationIds).toEqual([]);
	});

	it('a failed compare keeps only itself selected after the loose items were deleted', async () => {
		await historyStore.loadGenerations();
		historyStore.toggleGenerationSelection('solo');
		historyStore.toggleGenerationSelection('rep');
		mockDeleteGridById.mockRejectedValue(new Error('boom'));
		serve([repOne, repTwo]);

		await expect(historyStore.bulkDeleteGenerations()).rejects.toThrow('boom');

		expect(mockBulkDeleteGenerations).toHaveBeenCalledWith(['solo']);
		const state = get(historyStore);
		expect(state.generations.map((gen) => gen.id)).toEqual(['rep', 'rep2']);
		expect(state.selectedGenerationIds).toEqual(['rep']);
		expect(state.selectionMode).toBe(true);
	});

	it('with two compares where one fails, the other is gone and the failed one stays selected', async () => {
		await historyStore.loadGenerations();
		historyStore.toggleGenerationSelection('rep');
		historyStore.toggleGenerationSelection('rep2');
		mockDeleteGridById.mockImplementation((id: string) =>
			id === 'grid-2' ? Promise.reject(new Error('nope')) : Promise.resolve()
		);
		serve([solo, repTwo]);

		await expect(historyStore.bulkDeleteGenerations()).rejects.toThrow('nope');

		expect(mockDeleteGridById).toHaveBeenCalledTimes(2);
		expect(mockBulkDeleteGenerations).not.toHaveBeenCalled();
		const state = get(historyStore);
		expect(state.generations.map((gen) => gen.id)).toEqual(['solo', 'rep2']);
		expect(state.selectedGenerationIds).toEqual(['rep2']);
	});

	it('throws when the loose bulk delete reports failure and keeps those ids selected', async () => {
		await historyStore.loadGenerations();
		historyStore.toggleGenerationSelection('solo');
		mockBulkDeleteGenerations.mockResolvedValue({ success: false, message: 'denied' });

		await expect(historyStore.bulkDeleteGenerations()).rejects.toThrow('denied');

		expect(get(historyStore).selectedGenerationIds).toEqual(['solo']);
	});
});

describe('delete confirmation copy', () => {
	it('states the number of images for a compare', () => {
		expect(deleteCopy(items[1]).message).toContain('Delete this X/Y compare? 6 images will be deleted.');
		expect(gridDeleteMessage(1)).toContain('1 image will');
	});

	it('names the cell when a single compare cell is deleted', () => {
		const cell = { id: 'c', files: [], grid_id: 'grid-1', axis_values: { Steps: '20' } } as unknown as GenerationHistoryItem;
		const copy = deleteCopy(cell);
		expect(copy.title).toBe('Delete Compare Cell');
		expect(copy.message).toContain('Steps: 20');
	});

	it('counts compare images in a mixed selection', () => {
		const plan = planBulkDelete(['rep', 'solo'], items);
		expect(plan).toEqual({ gridIds: ['grid-1'], looseIds: ['solo'], imageCount: 7 });
		expect(bulkDeleteMessage(plan)).toContain('1 X/Y compare and 1 other generation(s)? 7 images will be deleted.');
	});
});
