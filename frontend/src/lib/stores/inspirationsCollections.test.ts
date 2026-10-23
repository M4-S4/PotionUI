import { beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';

vi.mock('$lib/services/api/index', () => ({
	api: {
		listInspirationCollections: vi.fn(),
		createInspirationCollection: vi.fn(),
		updateInspirationCollection: vi.fn(),
		deleteInspirationCollection: vi.fn(),
		addInspirationToCollection: vi.fn(),
		removeInspirationFromCollection: vi.fn()
	}
}));

vi.mock('$lib/utils/logger', () => ({
	logger: { error: vi.fn() },
	getErrorMessage: (error: unknown) => String(error)
}));

import { api } from '$lib/services/api/index';
import { inspirationsCollectionsStore } from './inspirationsCollections';

const list = vi.mocked(api.listInspirationCollections);

function deferred<T>() {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((res) => {
		resolve = res;
	});
	return { promise, resolve };
}

function listing(...ids: string[]) {
	return {
		success: true,
		data: { items: ids.map((id) => ({ id, name: id, item_count: 0 })) }
	} as never;
}

describe('inspirationsCollectionsStore', () => {
	beforeEach(() => {
		inspirationsCollectionsStore.reset();
		vi.resetAllMocks();
	});

	it('applies the most recently issued list even when an older one settles last', async () => {
		const first = deferred<unknown>();
		const second = deferred<unknown>();
		list.mockReturnValueOnce(first.promise as never).mockReturnValueOnce(second.promise as never);

		const firstLoad = inspirationsCollectionsStore.load();
		const secondLoad = inspirationsCollectionsStore.load();
		second.resolve(listing('fresh'));
		await secondLoad;
		first.resolve(listing('stale'));
		await firstLoad;

		expect(get(inspirationsCollectionsStore).collections.map((c) => c.id)).toEqual(['fresh']);
	});

	it('inserts a created collection before the reconciling refresh lands', async () => {
		const refresh = deferred<unknown>();
		vi.mocked(api.createInspirationCollection).mockResolvedValue({
			success: true,
			data: { collection: { id: 'made', name: 'made', item_count: 0 } }
		} as never);
		list.mockReturnValue(refresh.promise as never);

		const creating = inspirationsCollectionsStore.create('made', null);
		await Promise.resolve();
		await Promise.resolve();
		expect(get(inspirationsCollectionsStore).collections.map((c) => c.id)).toEqual(['made']);
		refresh.resolve(listing('made'));
		await creating;
	});

	it('has no bulk move and adds members one by one', async () => {
		expect('bulkMove' in inspirationsCollectionsStore).toBe(false);
		vi.mocked(api.addInspirationToCollection).mockResolvedValue({ success: true } as never);
		list.mockResolvedValue(listing());
		await inspirationsCollectionsStore.addMembers('c1', ['i1', 'i2']);
		expect(api.addInspirationToCollection).toHaveBeenCalledWith('c1', 'i1');
		expect(api.addInspirationToCollection).toHaveBeenCalledWith('c1', 'i2');
	});
});
