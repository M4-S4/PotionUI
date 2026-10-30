import { describe, expect, it } from 'vitest';
import { createPreviewKeeper, createVideoHandoff } from './videoHandoff';

const PREVIEW = 'data:image/png;base64,last-preview';
const FINAL = '/api/media/generations/gen-1/0.mp4';

function setup() {
	const states: boolean[] = [];
	const pending: Array<() => void> = [];
	const handoff = createVideoHandoff(
		(value) => states.push(value),
		(_url, onSettled) => pending.push(onSettled)
	);
	return { states, pending, handoff };
}

describe('createVideoHandoff', () => {
	it('holds the preview until the final video has loaded', () => {
		const { states, pending, handoff } = setup();
		handoff(FINAL, true);

		expect(states).toEqual([true]);

		pending[0]();

		expect(states).toEqual([true, false]);
	});

	it('does not hold when there was no preview to keep', () => {
		const { states, pending, handoff } = setup();
		handoff(FINAL, false);

		expect(states).toEqual([false]);
		expect(pending).toHaveLength(0);
	});

	it('probes a video once while it is still loading', () => {
		const { pending, handoff } = setup();
		handoff(FINAL, true);
		handoff(FINAL, true);
		handoff(FINAL, true);

		expect(pending).toHaveLength(1);
	});

	it('releases the hold when the video goes away before it loads', () => {
		const { states, pending, handoff } = setup();
		handoff(FINAL, true);
		handoff(null, true);
		pending[0]();

		expect(states).toEqual([true, false]);
	});

	it('does not hold again for a video it has already shown', () => {
		const { states, pending, handoff } = setup();
		handoff(FINAL, true);
		pending[0]();
		handoff(FINAL, true);

		expect(states).toEqual([true, false, false]);
		expect(pending).toHaveLength(1);
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
