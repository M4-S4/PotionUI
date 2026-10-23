import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ModelDigestRequestResult } from '$lib/services/api/models';
import { createDigestWatch, digestWatchDelay } from './modelDigestWatch';

type Listener = () => void;

function fakeDocument() {
	const listeners = new Map<string, Set<Listener>>();
	const doc = {
		hidden: false,
		addEventListener: (type: string, listener: Listener) => {
			if (!listeners.has(type)) listeners.set(type, new Set());
			listeners.get(type)!.add(listener);
		},
		removeEventListener: (type: string, listener: Listener) => {
			listeners.get(type)?.delete(listener);
		},
		dispatch: (type: string) => listeners.get(type)?.forEach((listener) => listener())
	};
	return doc;
}

let doc: ReturnType<typeof fakeDocument>;

beforeEach(() => {
	vi.useFakeTimers();
	doc = fakeDocument();
	vi.stubGlobal('document', doc);
});

afterEach(() => {
	vi.useRealTimers();
	vi.unstubAllGlobals();
});

function setup(results: Array<ModelDigestRequestResult | null | Error>, maxAttempts?: number) {
	const queue = [...results];
	const request = vi.fn(async (_retry: boolean) => {
		const next = queue.shift();
		if (next instanceof Error) throw next;
		return next ?? null;
	});
	const onProgress = vi.fn();
	const onReady = vi.fn();
	const onFailed = vi.fn();
	const watch = createDigestWatch({ request, onProgress, onReady, onFailed, maxAttempts });
	return { watch, request, onProgress, onReady, onFailed };
}

const QUEUED: ModelDigestRequestResult = { state: 'queued', eta_seconds: 5 };

describe('digestWatchDelay', () => {
	it('polls every 3 seconds when the estimate is unknown or short', () => {
		expect(digestWatchDelay(null)).toBe(3000);
		expect(digestWatchDelay(5)).toBe(3000);
	});

	it('slows down for long estimates and caps at 30 seconds', () => {
		expect(digestWatchDelay(100)).toBe(10000);
		expect(digestWatchDelay(86400)).toBe(30000);
	});
});

describe('createDigestWatch', () => {
	it('asks right away and reports the estimate while the file is queued', async () => {
		const { watch, request, onProgress } = setup([{ state: 'queued', eta_seconds: 240 }]);

		watch.start();
		await vi.advanceTimersByTimeAsync(0);

		expect(request).toHaveBeenCalledTimes(1);
		expect(onProgress).toHaveBeenCalledWith(240);
		watch.stop();
	});

	it('keeps asking until the file is ready, then stops', async () => {
		const { watch, request, onReady, onProgress } = setup([QUEUED, QUEUED, { state: 'ready', eta_seconds: null }]);

		watch.start();
		await vi.advanceTimersByTimeAsync(0);
		await vi.advanceTimersByTimeAsync(3000);
		await vi.advanceTimersByTimeAsync(3000);

		expect(request).toHaveBeenCalledTimes(3);
		expect(onProgress).toHaveBeenCalledTimes(2);
		expect(onReady).toHaveBeenCalledTimes(1);

		await vi.advanceTimersByTimeAsync(60000);
		expect(request).toHaveBeenCalledTimes(3);
	});

	it('waits as long as the latest estimate suggests before asking again', async () => {
		const { watch, request } = setup([{ state: 'queued', eta_seconds: 200 }, QUEUED]);

		watch.start();
		await vi.advanceTimersByTimeAsync(0);
		await vi.advanceTimersByTimeAsync(19999);
		expect(request).toHaveBeenCalledTimes(1);
		await vi.advanceTimersByTimeAsync(1);
		expect(request).toHaveBeenCalledTimes(2);
		watch.stop();
	});

	it('is ready at once when the file already has its digest', async () => {
		const { watch, onReady, onProgress } = setup([{ state: 'ready', eta_seconds: null }]);

		watch.start();
		await vi.advanceTimersByTimeAsync(0);

		expect(onReady).toHaveBeenCalledTimes(1);
		expect(onReady).toHaveBeenCalledWith(null);
		expect(onProgress).not.toHaveBeenCalled();
	});

	it('hands over the owner when the model was merged into another one', async () => {
		const { watch, onReady } = setup([{ state: 'ready', eta_seconds: 0, merged_into: 'owner-1' }]);

		watch.start();
		await vi.advanceTimersByTimeAsync(0);

		expect(onReady).toHaveBeenCalledWith('owner-1');
	});

	it('stops asking at once when every copy of the file failed', async () => {
		const { watch, request, onFailed, onReady, onProgress } = setup([QUEUED, { state: 'failed', eta_seconds: null }, QUEUED]);

		watch.start();
		await vi.advanceTimersByTimeAsync(0);
		await vi.advanceTimersByTimeAsync(3000);

		expect(onFailed).toHaveBeenCalledWith({ reason: 'failed' });
		expect(onReady).not.toHaveBeenCalled();
		expect(onProgress).toHaveBeenCalledTimes(1);

		await vi.advanceTimersByTimeAsync(120000);
		expect(request).toHaveBeenCalledTimes(2);
	});

	it('asks with retry only on the first ask of a retried start and resumes polling', async () => {
		const { watch, request, onReady } = setup([
			{ state: 'failed', eta_seconds: null },
			QUEUED,
			{ state: 'ready', eta_seconds: null }
		]);

		watch.start();
		await vi.advanceTimersByTimeAsync(0);
		watch.start({ retry: true });
		await vi.advanceTimersByTimeAsync(0);
		await vi.advanceTimersByTimeAsync(3000);

		expect(request.mock.calls.map((call) => call[0])).toEqual([false, true, false]);
		expect(onReady).toHaveBeenCalledTimes(1);
	});

	it('gives up with the error when the request throws and with no error when it answers nothing', async () => {
		const boom = new Error('boom');
		const failing = setup([boom]);
		failing.watch.start();
		await vi.advanceTimersByTimeAsync(0);
		expect(failing.onFailed).toHaveBeenCalledWith({ reason: 'request', error: boom });
		await vi.advanceTimersByTimeAsync(60000);
		expect(failing.request).toHaveBeenCalledTimes(1);

		const empty = setup([null]);
		empty.watch.start();
		await vi.advanceTimersByTimeAsync(0);
		expect(empty.onFailed).toHaveBeenCalledWith({ reason: 'request', error: undefined });
	});

	it('gives up with a timeout after the attempt cap and stops asking', async () => {
		const { watch, request, onFailed, onReady } = setup([QUEUED, QUEUED, QUEUED, QUEUED, QUEUED], 3);

		watch.start();
		await vi.advanceTimersByTimeAsync(0);
		await vi.advanceTimersByTimeAsync(3000);
		await vi.advanceTimersByTimeAsync(3000);
		expect(request).toHaveBeenCalledTimes(3);
		expect(onFailed).not.toHaveBeenCalled();

		await vi.advanceTimersByTimeAsync(3000);
		expect(onFailed).toHaveBeenCalledWith({ reason: 'timeout' });
		expect(onReady).not.toHaveBeenCalled();

		await vi.advanceTimersByTimeAsync(120000);
		expect(request).toHaveBeenCalledTimes(3);
	});

	it('does not ask while the tab is hidden and asks again as soon as it is visible', async () => {
		const { watch, request } = setup([QUEUED, QUEUED, QUEUED]);

		watch.start();
		await vi.advanceTimersByTimeAsync(0);
		expect(request).toHaveBeenCalledTimes(1);

		doc.hidden = true;
		await vi.advanceTimersByTimeAsync(60000);
		expect(request).toHaveBeenCalledTimes(1);

		doc.hidden = false;
		doc.dispatch('visibilitychange');
		await vi.advanceTimersByTimeAsync(0);
		expect(request).toHaveBeenCalledTimes(2);
		watch.stop();
	});

	it('stop cancels the pending poll and ignores an answer that arrives later', async () => {
		let release: (value: ModelDigestRequestResult) => void = () => {};
		const request = vi.fn(() => new Promise<ModelDigestRequestResult>((resolve) => (release = resolve)));
		const onReady = vi.fn();
		const watch = createDigestWatch({ request, onProgress: vi.fn(), onReady, onFailed: vi.fn() });

		watch.start();
		watch.stop();
		release({ state: 'ready', eta_seconds: null });
		await vi.advanceTimersByTimeAsync(0);

		expect(onReady).not.toHaveBeenCalled();
		await vi.advanceTimersByTimeAsync(60000);
		expect(request).toHaveBeenCalledTimes(1);
	});

	it('start again after stop begins a fresh watch', async () => {
		const { watch, request } = setup([QUEUED, QUEUED]);

		watch.start();
		await vi.advanceTimersByTimeAsync(0);
		watch.stop();
		watch.start();
		await vi.advanceTimersByTimeAsync(0);

		expect(request).toHaveBeenCalledTimes(2);
		watch.stop();
	});
});
