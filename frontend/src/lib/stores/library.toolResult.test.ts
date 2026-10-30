import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';

const mockApi = vi.hoisted(() => ({
	listLibraryItems: vi.fn(),
	getLibraryFacets: vi.fn(),
	getTags: vi.fn(),
	listCollections: vi.fn(),
	deleteLibraryItem: vi.fn(),
	uploadLibraryMedia: vi.fn()
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

	it('does not mark rows as placed when they were reloaded rather than inserted', async () => {
		mockApi.listLibraryItems.mockResolvedValue({
			success: true,
			data: { items: [row('n1'), row('a'), row('b')], total: 4, limit: 24, offset: 0 }
		});

		await libraryStore.applyToolResult([edited('n1')], ['elsewhere']);

		expect(get(libraryStore).freshIds).toEqual(['n1']);
		expect(get(libraryStore).placedIds).toEqual([]);
	});

	it('marks inline rows as placed so the grid files them next to their source', async () => {
		await libraryStore.applyToolResult([edited('n1')], ['b']);

		expect(get(libraryStore).placedIds).toEqual(['n1']);
	});
});

describe('New marks expire', () => {
	beforeEach(async () => {
		vi.clearAllMocks();
		libraryStore.reset();
		libraryCollectionsStore.reset();
		mockApi.getLibraryFacets.mockResolvedValue({ success: true, data: { media_types: {} } });
		mockApi.listCollections.mockResolvedValue({ success: true, data: { collections: [], smart_counts: null } });
		mockApi.deleteLibraryItem.mockResolvedValue({ success: true });
		await load([row('a'), row('b'), row('c')]);
		await libraryStore.applyToolResult([edited('n1')], ['b']);
		expect(get(libraryStore).freshIds).toEqual(['n1']);
	});

	it('when the page reloads', async () => {
		await load([row('n1'), row('a'), row('b')], 4);

		expect(get(libraryStore).freshIds).toEqual([]);
		expect(get(libraryStore).placedIds).toEqual([]);
	});

	it('when new rows are brought into view', async () => {
		await libraryStore.showNewRows();

		expect(get(libraryStore).freshIds).toEqual([]);
		expect(get(libraryStore).placedIds).toEqual([]);
	});

	it('when files are uploaded', async () => {
		mockApi.uploadLibraryMedia.mockResolvedValue({ success: true });

		await libraryStore.upload([new File(['x'], 'x.png', { type: 'image/png' })]);

		expect(get(libraryStore).freshIds).toEqual([]);
	});

	it('when an item is deleted', async () => {
		await libraryStore.deleteItem('a');

		expect(get(libraryStore).freshIds).toEqual([]);
	});

	it('when the library is cleared by clearFresh, which the page calls on mount and on leaving', () => {
		libraryStore.clearFresh();

		expect(get(libraryStore).freshIds).toEqual([]);
		expect(get(libraryStore).placedIds).toEqual([]);
	});

	it('but not when a reload fails, which brings in nothing that could replace them', async () => {
		mockApi.listLibraryItems.mockResolvedValue({ success: false });
		await libraryStore.load();

		expect(get(libraryStore).freshIds).toEqual(['n1']);
	});
});

describe('replace versus add', () => {
	beforeEach(async () => {
		vi.clearAllMocks();
		libraryStore.reset();
		libraryCollectionsStore.reset();
		mockApi.getLibraryFacets.mockResolvedValue({ success: true, data: { media_types: {} } });
		mockApi.listCollections.mockResolvedValue({ success: true, data: { collections: [], smart_counts: null } });
		await load([row('a'), row('b'), row('c')]);
	});

	it('a replace for a row that is no longer on the page changes nothing and stays on the page', async () => {
		libraryStore.setPage(3);
		mockApi.listLibraryItems.mockClear();

		const result = await libraryStore.applyToolResult([edited('gone')], ['gone'], { replaced: true });

		expect(result).toEqual({ added: 0, replaced: 0 });
		expect(mockApi.listLibraryItems).not.toHaveBeenCalled();
		expect(get(libraryStore).currentPage).toBe(3);
		expect(ids()).toEqual(['a', 'b', 'c']);
		expect(get(libraryStore).freshIds).toEqual([]);
	});

	it('a replace of a row on the page patches it where it stands', async () => {
		const result = await libraryStore.applyToolResult([edited('b', { url: '/u/b-new.mp4' })], ['b'], {
			replaced: true
		});

		expect(result).toEqual({ added: 0, replaced: 1 });
		expect(get(libraryStore).items[1].url).toBe('/u/b-new.mp4');
		expect(get(libraryStore).totalCount).toBe(3);
	});

	it('a replace still refreshes the counts', async () => {
		mockApi.getLibraryFacets.mockClear();

		await libraryStore.applyToolResult([edited('gone')], ['gone'], { replaced: true });

		expect(mockApi.getLibraryFacets).toHaveBeenCalledTimes(1);
	});

	it('a replace patches the open details item even when its row is on another page', async () => {
		libraryStore.setSelectedItem(row('far') as never);

		await libraryStore.applyToolResult([edited('far', { url: '/u/far-new.mp4' })], ['far'], {
			replaced: true
		});

		expect(get(libraryStore).selectedItem?.url).toBe('/u/far-new.mp4');
		expect(ids()).toEqual(['a', 'b', 'c']);
	});

	it('a result told to be new is added even if the loaded page has nothing with that source', async () => {
		const result = await libraryStore.applyToolResult([edited('n1')], ['b'], { replaced: false });

		expect(result).toEqual({ added: 1, replaced: 0 });
		expect(ids()).toEqual(['a', 'n1', 'b', 'c']);
	});

	it('never lists one row twice, even when a result told to be new carries an id the page holds', async () => {
		await libraryStore.applyToolResult([edited('b', { url: '/u/b-again.mp4' })], ['b'], { replaced: false });

		expect(ids()).toEqual(['a', 'b', 'c']);
		expect(get(libraryStore).items[1].url).toBe('/u/b-again.mp4');
	});

	it('still works out replace from the page when the caller does not say', async () => {
		expect(await libraryStore.applyToolResult([edited('b')], ['b'])).toEqual({ added: 0, replaced: 1 });
		expect(await libraryStore.applyToolResult([edited('n1')], ['b'])).toEqual({ added: 1, replaced: 0 });
	});
});

describe('with a filter on', () => {
	beforeEach(async () => {
		vi.clearAllMocks();
		libraryStore.reset();
		libraryCollectionsStore.reset();
		mockApi.getLibraryFacets.mockResolvedValue({ success: true, data: { media_types: {} } });
		mockApi.listCollections.mockResolvedValue({ success: true, data: { collections: [], smart_counts: null } });
		libraryStore.setFilter('mediaType', 'video');
		await load([row('a'), row('b'), row('c')]);
		mockApi.listLibraryItems.mockClear();
	});

	it('reloads the view instead of inserting a row into it', async () => {
		mockApi.listLibraryItems.mockResolvedValue({
			success: true,
			data: { items: [row('a'), row('b'), row('c')], total: 3, limit: 24, offset: 0 }
		});

		await libraryStore.applyToolResult([edited('n1', { media_type: 'image' })], ['b']);

		expect(mockApi.listLibraryItems).toHaveBeenCalledTimes(1);
		expect(mockApi.listLibraryItems).toHaveBeenCalledWith(expect.objectContaining({ media_type: 'video' }));
		expect(ids()).toEqual(['a', 'b', 'c']);
	});

	it('shows the server rows as they are, with no row built from the result', async () => {
		mockApi.listLibraryItems.mockResolvedValue({
			success: true,
			data: { items: [row('n1', { tags: [{ id: 't', name: 'keep' }], is_favorite: true }), row('a')], total: 2, limit: 24, offset: 0 }
		});

		await libraryStore.applyToolResult([edited('n1')], ['a']);

		const shown = get(libraryStore).items.find((item) => item.id === 'n1')!;
		expect(shown.tags).toEqual([{ id: 't', name: 'keep' }]);
		expect((shown as { is_favorite?: boolean }).is_favorite).toBe(true);
	});

	it('does not flash the loading state while it reloads', async () => {
		const seen: boolean[] = [];
		const stop = libraryStore.subscribe((state) => seen.push(state.loading));
		mockApi.listLibraryItems.mockResolvedValue({
			success: true,
			data: { items: [row('a')], total: 1, limit: 24, offset: 0 }
		});

		await libraryStore.applyToolResult([edited('n1')], ['a']);
		stop();

		expect(seen).not.toContain(true);
	});

	it('marks the new row only for the tag, not for day anchoring', async () => {
		mockApi.listLibraryItems.mockResolvedValue({
			success: true,
			data: { items: [row('n1'), row('a')], total: 2, limit: 24, offset: 0 }
		});

		await libraryStore.applyToolResult([edited('n1')], ['a']);

		expect(get(libraryStore).freshIds).toEqual(['n1']);
		expect(get(libraryStore).placedIds).toEqual([]);
	});

	it('leaves the filter and the page as they were', async () => {
		mockApi.listLibraryItems.mockResolvedValue({
			success: true,
			data: { items: [row('a')], total: 1, limit: 24, offset: 0 }
		});

		await libraryStore.applyToolResult([edited('n1')], ['a']);

		expect(get(libraryStore).filters.mediaType).toBe('video');
		expect(get(libraryStore).currentPage).toBe(1);
	});

	it('keeps a replace in place without reloading', async () => {
		await libraryStore.applyToolResult([edited('b', { url: '/u/b-new.mp4' })], ['b'], { replaced: true });

		expect(mockApi.listLibraryItems).not.toHaveBeenCalled();
		expect(get(libraryStore).items[1].url).toBe('/u/b-new.mp4');
	});

	it('inserts inline again once the filters are cleared', async () => {
		libraryStore.clearFilters();
		await load([row('a'), row('b'), row('c')]);
		mockApi.listLibraryItems.mockClear();

		await libraryStore.applyToolResult([edited('n1')], ['b']);

		expect(mockApi.listLibraryItems).not.toHaveBeenCalled();
		expect(ids()).toEqual(['a', 'n1', 'b', 'c']);
	});
});
