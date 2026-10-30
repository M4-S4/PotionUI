// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { restoreScroll } from '$lib/utils/restoreScroll';

let observed: (() => void) | null = null;
let originalObserver: unknown;

class CapturingObserver {
	constructor(callback: () => void) {
		observed = callback;
	}
	observe() {}
	unobserve() {}
	disconnect() {
		observed = null;
	}
}

function scroller(initialMax: number) {
	const node = document.createElement('div');
	node.appendChild(document.createElement('div'));
	const state = { max: initialMax, top: 0 };
	Object.defineProperty(node, 'scrollTop', {
		get: () => state.top,
		set: (value: number) => {
			state.top = Math.min(value, state.max);
		}
	});
	return { node, state };
}

beforeEach(() => {
	vi.useFakeTimers();
	originalObserver = (globalThis as { ResizeObserver?: unknown }).ResizeObserver;
	(globalThis as { ResizeObserver?: unknown }).ResizeObserver = CapturingObserver;
});

afterEach(() => {
	vi.useRealTimers();
	(globalThis as { ResizeObserver?: unknown }).ResizeObserver = originalObserver;
	observed = null;
});

describe('restoreScroll', () => {
	it('scrolls to the saved position at once when the content is already tall enough', () => {
		const { node } = scroller(1000);
		restoreScroll(node, 300);
		expect(node.scrollTop).toBe(300);
		expect(observed).toBeNull();
	});

	it('keeps trying as the content grows until the position is reachable', () => {
		const { node, state } = scroller(100);
		restoreScroll(node, 300);
		expect(node.scrollTop).toBe(100);

		state.max = 500;
		observed?.();

		expect(node.scrollTop).toBe(300);
		expect(observed).toBeNull();
	});

	it('stops once the user scrolls, so it never fights them', () => {
		const { node, state } = scroller(100);
		restoreScroll(node, 300);
		node.dispatchEvent(new Event('wheel'));

		state.max = 500;
		observed?.();

		expect(node.scrollTop).toBe(100);
	});

	it('gives up after two seconds', () => {
		const { node, state } = scroller(100);
		restoreScroll(node, 300);
		vi.advanceTimersByTime(2100);

		state.max = 500;
		observed?.();

		expect(node.scrollTop).toBe(100);
	});

	it('does nothing without a saved position', () => {
		const { node } = scroller(1000);
		restoreScroll(node, null);
		expect(node.scrollTop).toBe(0);
	});
});
