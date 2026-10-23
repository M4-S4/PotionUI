import { beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getTags: vi.fn(),
		createTag: vi.fn(),
		deleteTag: vi.fn()
	}
}));

import { api } from '$lib/services/api/index';
import { tagCatalog, TAG_CATALOG_TTL_MS } from './tagCatalog';

const getTags = vi.mocked(api.getTags);
const createTag = vi.mocked(api.createTag);
const deleteTag = vi.mocked(api.deleteTag);

function listing(...names: string[]) {
	return { success: true, data: { tags: names.map((name) => ({ id: name, name })) } } as never;
}

function deferred<T>() {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((res) => {
		resolve = res;
	});
	return { promise, resolve };
}

describe('tagCatalog', () => {
	beforeEach(() => {
		vi.useRealTimers();
		tagCatalog.reset();
		getTags.mockReset();
		createTag.mockReset();
		deleteTag.mockReset();
	});

	it('serves a second read from the cache and dedupes concurrent reads', async () => {
		getTags.mockResolvedValue(listing('a'));
		const [first, second] = await Promise.all([tagCatalog.get('MODEL'), tagCatalog.get('MODEL')]);
		await tagCatalog.get('MODEL');
		expect(getTags).toHaveBeenCalledTimes(1);
		expect(first).toBe(second);
	});

	it('keeps a separate list per tag type', async () => {
		getTags.mockImplementation(((type: string) => Promise.resolve(listing(type))) as never);
		expect((await tagCatalog.get('MODEL'))[0].name).toBe('MODEL');
		expect((await tagCatalog.get('GENERATION'))[0].name).toBe('GENERATION');
		expect(getTags).toHaveBeenCalledTimes(2);
	});

	it('refetches after the TTL and when fresh is requested', async () => {
		vi.useFakeTimers();
		getTags.mockResolvedValue(listing('a'));
		await tagCatalog.get('MODEL');
		await tagCatalog.get('MODEL', { fresh: true });
		expect(getTags).toHaveBeenCalledTimes(2);
		vi.advanceTimersByTime(TAG_CATALOG_TTL_MS + 1);
		await tagCatalog.get('MODEL');
		expect(getTags).toHaveBeenCalledTimes(3);
	});

	it('create shows the new tag to subscribers at once and forces the next read to refetch', async () => {
		getTags.mockResolvedValue(listing('a'));
		await tagCatalog.get('GENERATION');
		createTag.mockResolvedValue({ success: true, data: { tag: { id: 'b', name: 'b' } } } as never);
		await tagCatalog.create('b', 'GENERATION');
		expect(get(tagCatalog.tags('GENERATION')).map((t) => t.id)).toEqual(['a', 'b']);
		getTags.mockResolvedValue(listing('a', 'b'));
		await tagCatalog.get('GENERATION');
		expect(getTags).toHaveBeenCalledTimes(2);
	});

	it('remove drops the tag from subscribers and invalidates', async () => {
		getTags.mockResolvedValue(listing('a', 'b'));
		await tagCatalog.get('MODEL');
		deleteTag.mockResolvedValue({ success: true } as never);
		await tagCatalog.remove('a', 'MODEL');
		expect(get(tagCatalog.tags('MODEL')).map((t) => t.id)).toEqual(['b']);
		await tagCatalog.get('MODEL');
		expect(getTags).toHaveBeenCalledTimes(2);
	});

	it('a response that was in flight when the list was invalidated is not cached as fresh', async () => {
		const stale = deferred<unknown>();
		getTags.mockReturnValueOnce(stale.promise as never);
		const pending = tagCatalog.get('MODEL');
		tagCatalog.invalidate('MODEL');
		stale.resolve(listing('old'));
		await pending;
		getTags.mockResolvedValue(listing('new'));
		expect((await tagCatalog.get('MODEL'))[0].name).toBe('new');
	});

	it('does not cache a failed request', async () => {
		getTags.mockRejectedValueOnce(new Error('offline'));
		await expect(tagCatalog.get('MODEL')).rejects.toThrow('offline');
		getTags.mockResolvedValue(listing('a'));
		expect(await tagCatalog.get('MODEL')).toHaveLength(1);
	});

	it('aborting one waiter rejects only that waiter and keeps the shared request alive', async () => {
		const shared = deferred<unknown>();
		let sharedSignal!: AbortSignal;
		getTags.mockImplementation(((_type: string, signal: AbortSignal) => {
			sharedSignal = signal;
			return shared.promise;
		}) as never);
		const controller = new AbortController();
		const aborted = tagCatalog.get('MODEL', { signal: controller.signal });
		const other = tagCatalog.get('MODEL');
		controller.abort();
		await expect(aborted).rejects.toMatchObject({ code: 'ERR_CANCELED' });
		expect(sharedSignal.aborted).toBe(false);
		shared.resolve(listing('a'));
		expect(await other).toHaveLength(1);
		expect(getTags).toHaveBeenCalledTimes(1);
	});

	it('aborts the shared request once every waiter has left', async () => {
		let sharedSignal!: AbortSignal;
		getTags.mockImplementation(((_type: string, signal: AbortSignal) => {
			sharedSignal = signal;
			return new Promise(() => {});
		}) as never);
		const controller = new AbortController();
		const aborted = tagCatalog.get('MODEL', { signal: controller.signal });
		controller.abort();
		await expect(aborted).rejects.toBeTruthy();
		expect(sharedSignal.aborted).toBe(true);
	});
	it('a failed create or remove leaves subscribers and the cache untouched', async () => {
		getTags.mockResolvedValue(listing('a'));
		await tagCatalog.get('MODEL');
		createTag.mockResolvedValue({ success: false } as never);
		deleteTag.mockResolvedValue({ success: false } as never);
		await tagCatalog.create('b', 'MODEL');
		await tagCatalog.remove('a', 'MODEL');
		expect(get(tagCatalog.tags('MODEL')).map((t) => t.id)).toEqual(['a']);
		await tagCatalog.get('MODEL');
		expect(getTags).toHaveBeenCalledTimes(1);
	});

	it('a pre-aborted signal rejects without calling getTags', async () => {
		getTags.mockResolvedValue(listing('a'));
		const controller = new AbortController();
		controller.abort();
		await expect(tagCatalog.get('MODEL', { signal: controller.signal })).rejects.toMatchObject({
			code: 'ERR_CANCELED'
		});
		expect(getTags).not.toHaveBeenCalled();
	});

	it('reset during a pending request leaves the store empty once it settles', async () => {
		const pending = deferred<unknown>();
		getTags.mockReturnValueOnce(pending.promise as never);
		const read = tagCatalog.get('MODEL');
		tagCatalog.reset();
		pending.resolve(listing('late'));
		await read;
		expect(get(tagCatalog.tags('MODEL'))).toEqual([]);
	});
});
