import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createPreviewController, PREVIEW_DEBOUNCE_MS, type PreviewState } from './preview';

function setup(request: (signal: AbortSignal) => Promise<Blob>) {
	const states: PreviewState[] = [];
	let counter = 0;
	const revoked: string[] = [];
	const controller = createPreviewController({
		request,
		onChange: (state) => states.push(state),
		createUrl: () => `blob:${++counter}`,
		revokeUrl: (url) => revoked.push(url)
	});
	return { controller, states, revoked };
}

const blob = () => new Blob(['x'], { type: 'image/webp' });

beforeEach(() => {
	vi.useFakeTimers();
});

afterEach(() => {
	vi.useRealTimers();
});

describe('createPreviewController', () => {
	it('waits 300 ms after the last change before asking the server', async () => {
		const request = vi.fn().mockResolvedValue(blob());
		const { controller } = setup(request);

		controller.schedule();
		await vi.advanceTimersByTimeAsync(PREVIEW_DEBOUNCE_MS - 1);
		expect(request).not.toHaveBeenCalled();
		await vi.advanceTimersByTimeAsync(1);
		expect(request).toHaveBeenCalledTimes(1);
		expect(PREVIEW_DEBOUNCE_MS).toBe(300);
	});

	it('folds a burst of changes into one request', async () => {
		const request = vi.fn().mockResolvedValue(blob());
		const { controller } = setup(request);

		controller.schedule();
		await vi.advanceTimersByTimeAsync(200);
		controller.schedule();
		await vi.advanceTimersByTimeAsync(200);
		controller.schedule();
		await vi.advanceTimersByTimeAsync(299);
		expect(request).not.toHaveBeenCalled();
		await vi.advanceTimersByTimeAsync(1);

		expect(request).toHaveBeenCalledTimes(1);
	});

	it('can skip the wait for the first render', async () => {
		const request = vi.fn().mockResolvedValue(blob());
		const { controller } = setup(request);
		controller.schedule(true);
		await vi.advanceTimersByTimeAsync(0);
		expect(request).toHaveBeenCalledTimes(1);
	});

	it('reports loading, then the url of the rendered image', async () => {
		const { controller, states } = setup(vi.fn().mockResolvedValue(blob()));

		controller.schedule();
		expect(states.at(-1)).toEqual({ url: null, loading: true, error: null });
		await vi.advanceTimersByTimeAsync(PREVIEW_DEBOUNCE_MS);

		expect(states.at(-1)).toEqual({ url: 'blob:1', loading: false, error: null });
	});

	it('keeps showing the last image while the next one renders, then frees it', async () => {
		const { controller, states, revoked } = setup(vi.fn().mockResolvedValue(blob()));
		controller.schedule(true);
		await vi.advanceTimersByTimeAsync(0);

		controller.schedule();
		expect(states.at(-1)).toEqual({ url: 'blob:1', loading: true, error: null });
		expect(revoked).toEqual([]);
		await vi.advanceTimersByTimeAsync(PREVIEW_DEBOUNCE_MS);

		expect(states.at(-1)).toEqual({ url: 'blob:2', loading: false, error: null });
		expect(revoked).toEqual(['blob:1']);
	});

	it('aborts the request in flight when something newer is asked for, and ignores its answer', async () => {
		let resolveFirst: (value: Blob) => void = () => undefined;
		const signals: AbortSignal[] = [];
		const request = vi
			.fn()
			.mockImplementationOnce((signal: AbortSignal) => {
				signals.push(signal);
				return new Promise<Blob>((resolve) => (resolveFirst = resolve));
			})
			.mockImplementation((signal: AbortSignal) => {
				signals.push(signal);
				return Promise.resolve(blob());
			});
		const { controller, states } = setup(request);

		controller.schedule(true);
		await vi.advanceTimersByTimeAsync(0);
		controller.schedule();
		expect(signals[0].aborted).toBe(true);
		await vi.advanceTimersByTimeAsync(PREVIEW_DEBOUNCE_MS);
		resolveFirst(blob());
		await vi.advanceTimersByTimeAsync(0);

		expect(states.at(-1)).toEqual({ url: 'blob:1', loading: false, error: null });
		expect(states.filter((state) => state.url === 'blob:2')).toHaveLength(0);
	});

	it('says so when the server cannot render, without throwing', async () => {
		const { controller, states } = setup(vi.fn().mockRejectedValue(new Error('boom')));
		controller.schedule(true);
		await vi.advanceTimersByTimeAsync(0);
		expect(states.at(-1)).toEqual({ url: null, loading: false, error: 'The preview could not be rendered' });
	});

	it('does not call a cancelled request an error', async () => {
		const { controller, states } = setup(vi.fn().mockRejectedValue({ code: 'ERR_CANCELED' }));
		controller.schedule(true);
		await vi.advanceTimersByTimeAsync(0);
		expect(states.at(-1)?.error).toBeNull();
	});

	it('clear drops the image and any pending request', async () => {
		const request = vi.fn().mockResolvedValue(blob());
		const { controller, states, revoked } = setup(request);
		controller.schedule(true);
		await vi.advanceTimersByTimeAsync(0);

		controller.schedule();
		controller.clear();
		await vi.advanceTimersByTimeAsync(PREVIEW_DEBOUNCE_MS * 2);

		expect(request).toHaveBeenCalledTimes(1);
		expect(revoked).toEqual(['blob:1']);
		expect(states.at(-1)).toEqual({ url: null, loading: false, error: null });
	});

	it('destroy frees the image, aborts the request and leaves no timer', async () => {
		const signals: AbortSignal[] = [];
		const request = vi
			.fn()
			.mockResolvedValueOnce(blob())
			.mockImplementationOnce((signal: AbortSignal) => {
				signals.push(signal);
				return new Promise<Blob>(() => undefined);
			});
		const { controller, states, revoked } = setup(request);
		controller.schedule(true);
		await vi.advanceTimersByTimeAsync(0);
		controller.schedule(true);
		await vi.advanceTimersByTimeAsync(0);
		controller.schedule();
		const before = states.length;

		controller.destroy();

		expect(revoked).toEqual(['blob:1']);
		expect(signals[0].aborted).toBe(true);
		expect(vi.getTimerCount()).toBe(0);
		controller.schedule();
		await vi.advanceTimersByTimeAsync(PREVIEW_DEBOUNCE_MS);
		expect(states).toHaveLength(before);
	});
});
