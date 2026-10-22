// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { IndexingDigests, IndexingStatus } from '$lib/services/api/models';
import { digests } from '../../../tests/component/helpers/indexingFixtures';

vi.mock('$lib/services/api', () => ({
	api: {
		getIndexingStatus: vi.fn(),
		setIndexingSettings: vi.fn(),
		pauseIndexing: vi.fn(),
		resumeIndexing: vi.fn(),
		queueIndexingDigests: vi.fn()
	}
}));

function ok(data: IndexingStatus) {
	return { success: true, data };
}

function withDigests(overrides: Partial<IndexingDigests> = {}): IndexingStatus {
	return { state: 'done', found_on_disk: 10, digests: digests(overrides) };
}

const SAVED = { success: true, data: { state: 'done' } };

beforeEach(() => {
	vi.resetModules();
	vi.useFakeTimers();
});

afterEach(() => {
	vi.useRealTimers();
	vi.clearAllMocks();
});

describe('indexingStatusStore', () => {
	it('fetches once on first subscribe and does not poll while idle', async () => {
		const { api } = await import('$lib/services/api');
		vi.mocked(api.getIndexingStatus).mockResolvedValue(ok({ state: 'idle' }));
		const { indexingStatusStore } = await import('./indexingStatus.svelte');

		const unsubscribe = indexingStatusStore.subscribe();
		await indexingStatusStore.refresh();

		expect(api.getIndexingStatus).toHaveBeenCalledTimes(1);
		expect(indexingStatusStore.status?.state).toBe('idle');

		await vi.advanceTimersByTimeAsync(10_000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(1);

		unsubscribe();
	});

	it('polls every 2s while indexing and stops once the run finishes', async () => {
		const { api } = await import('$lib/services/api');
		vi.mocked(api.getIndexingStatus)
			.mockResolvedValueOnce(ok({ state: 'indexing', processed: 1, total: 10 }))
			.mockResolvedValueOnce(ok({ state: 'indexing', processed: 2, total: 10 }))
			.mockResolvedValueOnce(ok({ state: 'done', indexed: 10, found_on_disk: 10 }));
		const { indexingStatusStore } = await import('./indexingStatus.svelte');

		const unsubscribe = indexingStatusStore.subscribe();
		await indexingStatusStore.refresh();
		expect(indexingStatusStore.status?.state).toBe('indexing');

		await vi.advanceTimersByTimeAsync(2000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(2);
		expect(indexingStatusStore.status?.processed).toBe(2);

		await vi.advanceTimersByTimeAsync(2000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(3);
		expect(indexingStatusStore.status?.state).toBe('done');

		await vi.advanceTimersByTimeAsync(10_000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(3);

		unsubscribe();
	});

	it('keeps polling while restart_pending even if the reported state looks terminal', async () => {
		const { api } = await import('$lib/services/api');
		vi.mocked(api.getIndexingStatus).mockResolvedValue(
			ok({ state: 'done', indexed: 5, found_on_disk: 5, restart_pending: true })
		);
		const { indexingStatusStore } = await import('./indexingStatus.svelte');

		const unsubscribe = indexingStatusStore.subscribe();
		await indexingStatusStore.refresh();

		await vi.advanceTimersByTimeAsync(2000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(2);

		unsubscribe();
	});

	it('stops polling once the last subscriber unsubscribes', async () => {
		const { api } = await import('$lib/services/api');
		vi.mocked(api.getIndexingStatus).mockResolvedValue(ok({ state: 'indexing', processed: 0, total: 10 }));
		const { indexingStatusStore } = await import('./indexingStatus.svelte');

		const unsubscribe = indexingStatusStore.subscribe();
		await indexingStatusStore.refresh();
		unsubscribe();

		await vi.advanceTimersByTimeAsync(10_000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(1);
	});

	it('a second subscriber reuses the running poller instead of fetching again immediately', async () => {
		const { api } = await import('$lib/services/api');
		vi.mocked(api.getIndexingStatus).mockResolvedValue(ok({ state: 'indexing', processed: 0, total: 10 }));
		const { indexingStatusStore } = await import('./indexingStatus.svelte');

		const unsubscribeA = indexingStatusStore.subscribe();
		await indexingStatusStore.refresh();
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(1);

		const unsubscribeB = indexingStatusStore.subscribe();
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(1);

		unsubscribeA();
		await vi.advanceTimersByTimeAsync(2000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(2);

		unsubscribeB();
		await vi.advanceTimersByTimeAsync(2000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(2);
	});

	it('notifyRunStarted seeds the status immediately and starts polling without waiting for a fetch', async () => {
		const { api } = await import('$lib/services/api');
		vi.mocked(api.getIndexingStatus).mockResolvedValue(ok({ state: 'indexing', processed: 5, total: 10 }));
		const { indexingStatusStore } = await import('./indexingStatus.svelte');

		const unsubscribe = indexingStatusStore.subscribe();
		await indexingStatusStore.refresh();
		vi.mocked(api.getIndexingStatus).mockClear();

		indexingStatusStore.notifyRunStarted({ state: 'scanning' });
		expect(indexingStatusStore.status?.state).toBe('scanning');
		expect(api.getIndexingStatus).not.toHaveBeenCalled();

		await vi.advanceTimersByTimeAsync(2000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(1);

		unsubscribe();
	});

	it('keeps polling while files are queued after the scan is done and stops when the queue drains', async () => {
		const { api } = await import('$lib/services/api');
		vi.mocked(api.getIndexingStatus)
			.mockResolvedValueOnce(ok(withDigests({ queued: 2, pending_total: 2 })))
			.mockResolvedValueOnce(ok(withDigests({ queued: 1, pending_total: 1 })))
			.mockResolvedValue(ok(withDigests({ ready: 2 })));
		const { indexingStatusStore } = await import('./indexingStatus.svelte');

		const unsubscribe = indexingStatusStore.subscribe();
		await indexingStatusStore.refresh();

		await vi.advanceTimersByTimeAsync(2000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(2);
		await vi.advanceTimersByTimeAsync(2000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(3);

		await vi.advanceTimersByTimeAsync(10_000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(3);
		unsubscribe();
	});

	it('does not poll while the user has paused the queue', async () => {
		const { api } = await import('$lib/services/api');
		vi.mocked(api.getIndexingStatus).mockResolvedValue(ok(withDigests({ queued: 2, pending_total: 2, paused: true })));
		const { indexingStatusStore } = await import('./indexingStatus.svelte');

		const unsubscribe = indexingStatusStore.subscribe();
		await indexingStatusStore.refresh();
		await vi.advanceTimersByTimeAsync(10_000);

		expect(api.getIndexingStatus).toHaveBeenCalledTimes(1);
		unsubscribe();
	});

	it('keeps polling while a choice is awaited', async () => {
		const { api } = await import('$lib/services/api');
		const plan = { awaiting_choice: true, files: 30, bytes: 1, min_files: 20, min_bytes: 1 };
		vi.mocked(api.getIndexingStatus).mockResolvedValue(ok(withDigests({ plan, pending_total: 30, deferred: 30 })));
		const { indexingStatusStore } = await import('./indexingStatus.svelte');

		const unsubscribe = indexingStatusStore.subscribe();
		await indexingStatusStore.refresh();
		await vi.advanceTimersByTimeAsync(4000);

		expect(api.getIndexingStatus).toHaveBeenCalledTimes(3);
		unsubscribe();
	});

	it('a late subscriber refetches when the last status is stale and reuses a fresh one', async () => {
		const { api } = await import('$lib/services/api');
		vi.mocked(api.getIndexingStatus).mockResolvedValue(ok({ state: 'idle' }));
		const { indexingStatusStore } = await import('./indexingStatus.svelte');

		const first = indexingStatusStore.subscribe();
		await indexingStatusStore.refresh();
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(1);

		const fresh = indexingStatusStore.subscribe();
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(1);
		fresh();

		await vi.advanceTimersByTimeAsync(6000);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(1);
		const stale = indexingStatusStore.subscribe();
		await vi.advanceTimersByTimeAsync(0);
		expect(api.getIndexingStatus).toHaveBeenCalledTimes(2);

		stale();
		first();
	});

	describe('changes', () => {
		async function setup(status: IndexingStatus = withDigests({ ready: 5 })) {
			const { api } = await import('$lib/services/api');
			vi.mocked(api.getIndexingStatus).mockResolvedValue(ok(status));
			vi.mocked(api.setIndexingSettings).mockResolvedValue(SAVED as never);
			vi.mocked(api.pauseIndexing).mockResolvedValue(SAVED as never);
			vi.mocked(api.resumeIndexing).mockResolvedValue(SAVED as never);
			vi.mocked(api.queueIndexingDigests).mockResolvedValue(SAVED as never);
			const { indexingStatusStore } = await import('./indexingStatus.svelte');
			const unsubscribe = indexingStatusStore.subscribe();
			await indexingStatusStore.refresh();
			vi.mocked(api.getIndexingStatus).mockClear();
			return { api, store: indexingStatusStore, unsubscribe };
		}

		it('setLevel saves the level and refetches the status', async () => {
			const { api, store, unsubscribe } = await setup();

			expect(await store.setLevel('max')).toBe(true);

			expect(api.setIndexingSettings).toHaveBeenCalledWith({ level: 'max' });
			expect(api.getIndexingStatus).toHaveBeenCalledTimes(1);
			expect(store.busy).toBe(false);
			unsubscribe();
		});

		it('pause and resume call their routes', async () => {
			const { api, store, unsubscribe } = await setup();

			await store.pause();
			await store.resume();

			expect(api.pauseIndexing).toHaveBeenCalledTimes(1);
			expect(api.resumeIndexing).toHaveBeenCalledTimes(1);
			unsubscribe();
		});

		it('indexAll sets the background policy before queueing everything', async () => {
			const { api, store, unsubscribe } = await setup();
			const order: string[] = [];
			vi.mocked(api.setIndexingSettings).mockImplementation(async () => {
				order.push('policy');
				return SAVED as never;
			});
			vi.mocked(api.queueIndexingDigests).mockImplementation(async () => {
				order.push('queue');
				return SAVED as never;
			});

			await store.indexAll();

			expect(api.setIndexingSettings).toHaveBeenCalledWith({ policy: 'background' });
			expect(api.queueIndexingDigests).toHaveBeenCalledWith({ mode: 'all' });
			expect(order).toEqual(['policy', 'queue']);
			unsubscribe();
		});

		it('indexSelected queues every request first and moves an unanswered policy to on_demand last', async () => {
			const plan = { awaiting_choice: true, files: 30, bytes: 1, min_files: 20, min_bytes: 1 };
			const { api, store, unsubscribe } = await setup(withDigests({ policy: 'unset', plan, pending_total: 30 }));
			const order: string[] = [];
			vi.mocked(api.queueIndexingDigests).mockImplementation(async (request) => {
				order.push(`queue ${request.mode}`);
				return SAVED as never;
			});
			vi.mocked(api.setIndexingSettings).mockImplementation(async () => {
				order.push('policy');
				return SAVED as never;
			});

			await store.indexSelected([{ mode: 'matching', query: 'xl' }, { mode: 'selected', model_ids: ['a'] }]);

			expect(api.setIndexingSettings).toHaveBeenCalledWith({ policy: 'on_demand' });
			expect(api.queueIndexingDigests).toHaveBeenNthCalledWith(1, { mode: 'matching', query: 'xl' });
			expect(api.queueIndexingDigests).toHaveBeenNthCalledWith(2, { mode: 'selected', model_ids: ['a'] });
			expect(order).toEqual(['queue matching', 'queue selected', 'policy']);
			unsubscribe();
		});

		it('indexSelected leaves the policy unanswered when queueing fails', async () => {
			const plan = { awaiting_choice: true, files: 30, bytes: 1, min_files: 20, min_bytes: 1 };
			const { api, store, unsubscribe } = await setup(withDigests({ policy: 'unset', plan, pending_total: 30 }));
			vi.mocked(api.queueIndexingDigests).mockResolvedValue({ success: false, message: 'Queue is full' } as never);

			expect(await store.indexSelected([{ mode: 'selected', model_ids: ['a'] }])).toBe(false);

			expect(api.setIndexingSettings).not.toHaveBeenCalled();
			expect(store.actionError).toBe('Queue is full');
			unsubscribe();
		});

		it('indexSelected leaves an answered policy alone', async () => {
			const { api, store, unsubscribe } = await setup(withDigests({ policy: 'background', pending_total: 3 }));

			await store.indexSelected([{ mode: 'selected', model_ids: ['a'] }]);

			expect(api.setIndexingSettings).not.toHaveBeenCalled();
			expect(api.queueIndexingDigests).toHaveBeenCalledTimes(1);
			unsubscribe();
		});

		it('later sets the on_demand policy', async () => {
			const { api, store, unsubscribe } = await setup();

			await store.later();

			expect(api.setIndexingSettings).toHaveBeenCalledWith({ policy: 'on_demand' });
			unsubscribe();
		});

		it('stops at the first failing step, reports it and still refetches', async () => {
			const { api, store, unsubscribe } = await setup();
			vi.mocked(api.setIndexingSettings).mockResolvedValue({ success: false, message: 'No way' } as never);

			expect(await store.indexAll()).toBe(false);

			expect(api.queueIndexingDigests).not.toHaveBeenCalled();
			expect(store.actionError).toBe('No way');
			expect(api.getIndexingStatus).toHaveBeenCalledTimes(1);
			expect(store.busy).toBe(false);
			unsubscribe();
		});

		it('turns a thrown request error into the action error', async () => {
			const { api, store, unsubscribe } = await setup();
			vi.mocked(api.pauseIndexing).mockRejectedValue({ response: { data: { message: 'Forbidden' } } });

			expect(await store.pause()).toBe(false);

			expect(store.actionError).toBe('Forbidden');
			unsubscribe();
		});

		it('refetches after a change even when a poll is already in flight', async () => {
			const { api, store, unsubscribe } = await setup();
			let release: (value: unknown) => void = () => {};
			vi.mocked(api.getIndexingStatus).mockImplementationOnce(
				() => new Promise((resolve) => (release = resolve)) as never
			);
			const pending = store.refresh();

			const change = store.pause();
			await Promise.resolve();
			release(ok(withDigests({ ready: 5 })));
			await pending;
			await change;

			expect(api.getIndexingStatus).toHaveBeenCalledTimes(2);
			unsubscribe();
		});

		it('remembers a pick request until it is closed', async () => {
			const { store, unsubscribe } = await setup();

			store.requestPick();
			expect(store.pickRequested).toBe(true);
			store.closePick();
			expect(store.pickRequested).toBe(false);
			unsubscribe();
		});
	});
});
