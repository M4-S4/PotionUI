import { describe, it, expect, vi } from 'vitest';
import { buildSmartViews, createFilterSelection, createMembershipActions } from './scopeActions';

function setup(overrides: Partial<Parameters<typeof createMembershipActions>[0]> = {}) {
	const flash = vi.fn();
	const clearSelection = vi.fn();
	const add = vi.fn().mockResolvedValue({ success: true, data: { added: 2 } });
	const remove = vi.fn().mockResolvedValue({ success: true, data: { removed: 1 } });
	const create = vi.fn().mockResolvedValue({ success: true, data: { collection: { id: 'new' } } });
	const reloadCollections = vi.fn().mockResolvedValue(undefined);
	const afterRemove = vi.fn().mockResolvedValue(undefined);
	const actions = createMembershipActions({
		getSelectedIds: () => ['a', 'b', 'c'],
		add,
		remove,
		create,
		reloadCollections,
		clearSelection,
		afterRemove,
		flash,
		...overrides
	});
	return { actions, flash, clearSelection, add, remove, create, reloadCollections, afterRemove };
}

describe('createMembershipActions', () => {
	it('adds the selection, reloads collections, clears and reports the server count', async () => {
		const { actions, add, reloadCollections, clearSelection, flash } = setup();
		expect(await actions.addToCollection('c1')).toBe(true);
		expect(add).toHaveBeenCalledWith('c1', ['a', 'b', 'c']);
		expect(reloadCollections).toHaveBeenCalled();
		expect(clearSelection).toHaveBeenCalled();
		expect(flash).toHaveBeenCalledWith('Added 2 to collection');
	});

	it('does nothing for an empty selection', async () => {
		const { actions, add } = setup({ getSelectedIds: () => [] });
		expect(await actions.addToCollection('c1')).toBe(false);
		expect(add).not.toHaveBeenCalled();
	});

	it('keeps the selection when the add is rejected', async () => {
		const { actions, clearSelection, flash } = setup({
			add: vi.fn().mockResolvedValue({ success: false })
		});
		expect(await actions.addToCollection('c1')).toBe(false);
		expect(clearSelection).not.toHaveBeenCalled();
		expect(flash).toHaveBeenCalledWith('Failed to add to collection');
	});

	it('reports a thrown add as a failure', async () => {
		const { actions, flash } = setup({ add: vi.fn().mockRejectedValue(new Error('boom')) });
		expect(await actions.addToCollection('c1')).toBe(false);
		expect(flash).toHaveBeenCalledWith('Failed to add to collection');
	});

	it('creates a collection then adds the selection to it', async () => {
		const { actions, add } = setup();
		expect(await actions.createAndAdd('Fresh')).toBe(true);
		expect(add).toHaveBeenCalledWith('new', ['a', 'b', 'c']);
	});

	it('flashes and skips the add when creation yields no collection', async () => {
		const { actions, add, flash } = setup({ create: vi.fn().mockResolvedValue({ success: false }) });
		expect(await actions.createAndAdd('Fresh')).toBe(false);
		expect(add).not.toHaveBeenCalled();
		expect(flash).toHaveBeenCalledWith('Failed to create collection');
	});

	it('orders add as reload, clear, flash', async () => {
		const order: string[] = [];
		const { actions } = setup({
			reloadCollections: async () => {
				order.push('reload');
			},
			clearSelection: () => order.push('clear'),
			flash: () => order.push('flash')
		});
		await actions.addToCollection('c1');
		expect(order).toEqual(['reload', 'clear', 'flash']);
	});

	it('orders remove as reload, clear, flash, after', async () => {
		const order: string[] = [];
		const { actions } = setup({
			reloadCollections: async () => {
				order.push('reload');
			},
			clearSelection: () => order.push('clear'),
			flash: () => order.push('flash'),
			afterRemove: async () => {
				order.push('after');
			}
		});
		await actions.removeFromCollection('c1');
		expect(order).toEqual(['reload', 'clear', 'flash', 'after']);
	});

	it.each([
		['is rejected', () => vi.fn().mockResolvedValue({ success: false })],
		['throws', () => vi.fn().mockRejectedValue(new Error('boom'))]
	])('keeps the selection and skips afterRemove when remove %s', async (_name, makeRemove) => {
		const { actions, clearSelection, afterRemove, flash } = setup({ remove: makeRemove() });
		await actions.removeFromCollection('c1');
		expect(clearSelection).not.toHaveBeenCalled();
		expect(afterRemove).not.toHaveBeenCalled();
		expect(flash).toHaveBeenCalledWith('Failed to remove from collection');
	});

	it('returns false and flashes when create throws', async () => {
		const { actions, add, flash } = setup({ create: vi.fn().mockRejectedValue(new Error('boom')) });
		expect(await actions.createAndAdd('Fresh')).toBe(false);
		expect(add).not.toHaveBeenCalled();
		expect(flash).toHaveBeenCalledWith('Failed to create collection');
	});

	it('ignores remove without an active collection or remove handler', async () => {
		const first = setup();
		await first.actions.removeFromCollection(undefined);
		expect(first.remove).not.toHaveBeenCalled();
		const second = setup({ remove: undefined });
		await expect(second.actions.removeFromCollection('c1')).resolves.toBeUndefined();
	});
});

describe('buildSmartViews', () => {
	it('maps state, counts and handlers onto the three views', () => {
		const onAll = vi.fn();
		const onFavorites = vi.fn();
		const onUnsorted = vi.fn();
		const views = buildSmartViews({
			icon: 'photo',
			label: 'All uploads',
			isAll: false,
			favoritesOnly: true,
			unsorted: false,
			counts: { all: 9, favorites: 3, unsorted: 1 },
			onAll,
			onFavorites,
			onUnsorted
		});
		expect(views.allView).toMatchObject({ icon: 'photo', label: 'All uploads', active: false, count: 9 });
		expect(views.favoritesView).toMatchObject({ active: true, count: 3 });
		expect(views.unsortedView).toMatchObject({ active: false, count: 1 });
		views.favoritesView.onSelect();
		expect(onFavorites).toHaveBeenCalled();
		expect(onAll).not.toHaveBeenCalled();
	});
});

describe('createFilterSelection', () => {
	it('resets the unselected smart filters and reloads once', async () => {
		const setFilter = vi.fn();
		const reload = vi.fn().mockResolvedValue(undefined);
		const { applySelection, setDirectOnly } = createFilterSelection(setFilter, reload);
		await applySelection({ favoritesOnly: true });
		expect(setFilter.mock.calls).toEqual([
			['collectionId', undefined],
			['favoritesOnly', true],
			['unsorted', false]
		]);
		expect(reload).toHaveBeenCalledTimes(1);
		await setDirectOnly(true);
		expect(setFilter).toHaveBeenLastCalledWith('directOnly', true);
		expect(reload).toHaveBeenCalledTimes(2);
	});
});
