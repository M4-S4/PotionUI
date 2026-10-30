import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createPreviewKeeper, createVideoHandoff } from './videoHandoff';

const PREVIEW = 'data:image/png;base64,last-preview';
const FINAL = '/api/media/generations/gen-1/0.mp4';

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

function setup(timeoutMs = 1000) {
	const states: boolean[] = [];
	const handoff = createVideoHandoff((value) => states.push(value), timeoutMs);
	return { states, handoff };
}

describe('createVideoHandoff', () => {
	it('holds the preview until the visible player reports its first frame', () => {
		const { states, handoff } = setup();
		handoff.want(FINAL, true);

		expect(states).toEqual([true]);

		handoff.ready();

		expect(states).toEqual([true, false]);
	});

	it('does not hold when there was no preview to keep', () => {
		const { states, handoff } = setup();
		handoff.want(FINAL, false);

		expect(states).toEqual([false]);
	});

	it('keeps holding while the same video is asked for again', () => {
		const { states, handoff } = setup();
		handoff.want(FINAL, true);
		handoff.want(FINAL, true);
		handoff.want(FINAL, true);

		expect(states).toEqual([true]);
	});

	it('releases the hold when the video goes away', () => {
		const { states, handoff } = setup();
		handoff.want(FINAL, true);
		handoff.want(null, true);

		expect(states).toEqual([true, false]);
	});

	it('does not hold again for a video it has already shown', () => {
		const { states, handoff } = setup();
		handoff.want(FINAL, true);
		handoff.ready();
		handoff.want(FINAL, true);

		expect(states).toEqual([true, false]);
	});

	it('gives up after the timeout so a player that never paints cannot hold the preview forever', () => {
		const { states, handoff } = setup(1000);
		handoff.want(FINAL, true);

		vi.advanceTimersByTime(999);
		expect(states).toEqual([true]);

		vi.advanceTimersByTime(1);
		expect(states).toEqual([true, false]);
	});
});

describe('createPreviewKeeper', () => {
	it('keeps the last inline preview after the live preview clears', () => {
		const keep = createPreviewKeeper();

		expect(keep('gen-1', PREVIEW)).toBe(PREVIEW);
		expect(keep('gen-1', null)).toBe(PREVIEW);
		expect(keep('gen-1', FINAL)).toBe(PREVIEW);
	});

	it('forgets the preview when another generation starts', () => {
		const keep = createPreviewKeeper();
		keep('gen-1', PREVIEW);

		expect(keep('gen-2', null)).toBeNull();
	});
});
