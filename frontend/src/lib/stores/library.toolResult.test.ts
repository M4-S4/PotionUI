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
import { libraryCollectionsStore } from './collections';
import type { EditedMediaItem } from '$lib/services/api/media';

function row(id: string, extra: Record<string, unknown> = {}) {
	return {
		id,
		filename: `${id}.mp4`,
		original_filename: `${id}.mp4`,
		media_type: 'video',
		url: `/api/media/uploads/${id}.mp4`,
		thumbnail_medium: `/thumb/${id}.jpg`,
		tags: [{ id: 't1', name: 'keep' }],
		...extra
	};
}

function edited(id: string, extra: Partial<EditedMediaItem> = {}): EditedMediaItem {
	return {
		id,
		filename: `${id}-uuid.mp4`,
		original_filename: `${id}.mp4`,
		media_type: 'video',
		url: `/api/media/uploads/${id}-uuid.mp4`,
		width: 1920,
		height: 1080,
		created_at: '2026-10-29T10:00:00Z',
		...extra
	};
}

const ids = () => get(libraryStore).items.map((item) => item.id);

async function load(items: ReturnType<typeof row>[], total = items.length) {
	mockApi.listLibraryItems.mockResolvedValue({ success: true, data: { items, total, limit: 24, offset: 0 } });
	await libraryStore.load();
}

describe('library tool results', () => {
	beforeEach(async () => {
		vi.clearAllMocks();
		libraryStore.reset();
		libraryCollectionsStore.reset();
		mockApi.getLibraryFacets.mockResolvedValue({ success: true, data: { media_types: { video: 3 } } });
		mockApi.listCollections.mockResolvedValue({
			success: true,
			data: { collections: [], smart_counts: { all: 3, favorites: 0, unsorted: 3 } }
		});
		await load([row('a'), row('b'), row('c')]);
	});

	it('puts a new item next to the source it came from', async () => {
		await libraryStore.applyToolResult([edited('n1')], ['b']);

		expect(ids()).toEqual(['a', 'n1', 'b', 'c']);
	});

	it('keeps the order of several results in front of the source', async () => {
		await libraryStore.applyToolResult([edited('p1'), edited('p2'), edited('p3')], ['c']);

		expect(ids()).toEqual(['a', 'b', 'p1', 'p2', 'p3', 'c']);
	});

	it('inserts a joined result in front of the first of its sources on the page', async () => {
		await libraryStore.applyToolResult([edited('joined')], ['c', 'a']);

		expect(ids()).toEqual(['joined', 'a', 'b', 'c']);
	});

	it('marks what it added as new and nothing else', async () => {
		await libraryStore.applyToolResult([edited('n1')], ['b']);

		expect(get(libraryStore).freshIds).toEqual(['n1']);
	});

	it('shows the new row as a complete library item the card can draw', async () => {
		await libraryStore.applyToolResult([edited('n1')], ['b']);

		const added = get(libraryStore).items.find((item) => item.id === 'n1')!;
		expect(added).toMatchObject({
			filename: 'n1-uuid.mp4',
			original_filename: 'n1.mp4',
			url: '/api/media/uploads/n1-uuid.mp4',
			width: 1920,
			tags: [],
			is_favorite: false
		});
	});

	it('grows the total by what it added', async () => {
		await libraryStore.applyToolResult([edited('n1')], ['b']);

		expect(get(libraryStore).totalCount).toBe(4);
	});

	it('reports what it added and replaced', async () => {
		expect(await libraryStore.applyToolResult([edited('n1')], ['b'])).toEqual({ added: 1, replaced: 0 });
	});

	it('refreshes the type counts and the sidebar counts after adding', async () => {
		mockApi.getLibraryFacets.mockClear();
		mockApi.listCollections.mockClear();

		await libraryStore.applyToolResult([edited('n1')], ['b']);

		expect(mockApi.getLibraryFacets).toHaveBeenCalledTimes(1);
		expect(mockApi.listCollections).toHaveBeenCalledTimes(1);
	});

	it('patches a replaced row where it stands, keeping its tags, and adds nothing', async () => {
		const replaced = edited('b', { filename: 'b-new.mp4', url: '/api/media/uploads/b-new.mp4', width: 640 });

		const result = await libraryStore.applyToolResult([replaced], ['b']);

		expect(result).toEqual({ added: 0, replaced: 1 });
		expect(ids()).toEqual(['a', 'b', 'c']);
		const patched = get(libraryStore).items[1];
		expect(patched.url).toBe('/api/media/uploads/b-new.mp4');
		expect(patched.width).toBe(640);
		expect(patched.tags).toEqual([{ id: 't1', name: 'keep' }]);
		expect(get(libraryStore).freshIds).toEqual([]);
		expect(get(libraryStore).totalCount).toBe(3);
	});

	it('patches the open details item when its row was replaced', async () => {
		libraryStore.setSelectedItem(get(libraryStore).items[1]);

		await libraryStore.applyToolResult([edited('b', { url: '/api/media/uploads/b-new.mp4' })], ['b']);

		expect(get(libraryStore).selectedItem?.url).toBe('/api/media/uploads/b-new.mp4');
	});

	it('falls back to reloading the first page when the source is not on this page', async () => {
		mockApi.listLibraryItems.mockClear();
		mockApi.listLibraryItems.mockResolvedValue({
			success: true,
			data: { items: [row('n1'), row('a'), row('b')], total: 4, limit: 24, offset: 0 }
		});

		await libraryStore.applyToolResult([edited('n1')], ['elsewhere']);

		expect(mockApi.listLibraryItems).toHaveBeenCalledTimes(1);
		expect(ids()).toEqual(['n1', 'a', 'b']);
		expect(get(libraryStore).freshIds).toEqual(['n1']);
	});

	it('does not reload when the source is on the page', async () => {
		mockApi.listLibraryItems.mockClear();

		await libraryStore.applyToolResult([edited('n1')], ['a']);

		expect(mockApi.listLibraryItems).not.toHaveBeenCalled();
	});

	it('forgets which rows are new once the view changes', async () => {
		await libraryStore.applyToolResult([edited('n1')], ['b']);

		libraryStore.setPage(2);
		expect(get(libraryStore).freshIds).toEqual([]);

		await libraryStore.applyToolResult([edited('n2')], ['a']);
		libraryStore.setFilter('mediaType', 'video');
		expect(get(libraryStore).freshIds).toEqual([]);

		await libraryStore.applyToolResult([edited('n3')], ['a']);
		libraryStore.toggleTagFilter('t1');
		expect(get(libraryStore).freshIds).toEqual([]);

		await libraryStore.applyToolResult([edited('n4')], ['a']);
		libraryStore.clearFilters();
		expect(get(libraryStore).freshIds).toEqual([]);
	});

	it('can drop the new marks on its own', async () => {
		await libraryStore.applyToolResult([edited('n1')], ['b']);

		libraryStore.clearFresh();

		expect(get(libraryStore).freshIds).toEqual([]);
	});
});
