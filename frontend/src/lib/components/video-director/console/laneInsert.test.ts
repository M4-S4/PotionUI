import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { beginLaneDrag, insertFraction, isOverSelector, laneFractionFromClientX } from './laneInsert';

function lane(left: number, width: number): Element {
	return { getBoundingClientRect: () => ({ left, width }) } as unknown as Element;
}

function target(matches: boolean): HTMLElement {
	return { closest: () => (matches ? {} : null) } as unknown as HTMLElement;
}

function pointerDown() {
	return {
		stopPropagation: vi.fn(),
		preventDefault: vi.fn(),
		currentTarget: {},
		pointerId: 1
	} as unknown as PointerEvent & { stopPropagation: ReturnType<typeof vi.fn>; preventDefault: ReturnType<typeof vi.fn> };
}

const fakeWindow = new EventTarget();

function dispatch(type: string, init: Record<string, unknown> = {}) {
	const event = new Event(type, { cancelable: true });
	Object.assign(event, init);
	fakeWindow.dispatchEvent(event);
}

beforeEach(() => {
	vi.stubGlobal('window', fakeWindow);
});

afterEach(() => {
	dispatch('pointercancel');
	vi.unstubAllGlobals();
});

describe('laneFractionFromClientX', () => {
	it('maps a client x to a fraction of the lane width', () => {
		expect(laneFractionFromClientX(lane(100, 200), 200)).toBe(0.5);
	});

	it('clamps outside the lane', () => {
		expect(laneFractionFromClientX(lane(100, 200), 0)).toBe(0);
		expect(laneFractionFromClientX(lane(100, 200), 900)).toBe(1);
	});

	it('returns null without a lane or with no width', () => {
		expect(laneFractionFromClientX(undefined, 10)).toBeNull();
		expect(laneFractionFromClientX(lane(0, 0), 10)).toBeNull();
	});
});

describe('insertFraction', () => {
	it('returns null when the pointer is over a mark', () => {
		const e = { target: target(true), clientX: 150 } as unknown as MouseEvent;
		expect(insertFraction(e, lane(100, 200), '.mark')).toBeNull();
		expect(isOverSelector(e, '.mark')).toBe(true);
	});

	it('returns the lane fraction over empty lane', () => {
		const e = { target: target(false), clientX: 150 } as unknown as MouseEvent;
		expect(insertFraction(e, lane(100, 200), '.mark')).toBe(0.25);
	});
});

describe('beginLaneDrag', () => {
	it('does not start without a lane but still swallows the event', () => {
		const e = pointerDown();
		const handlers = { onMove: vi.fn(), onCommit: vi.fn(), onCancel: vi.fn() };
		expect(beginLaneDrag(e, undefined, handlers)).toBe(false);
		expect(e.stopPropagation).toHaveBeenCalled();
		dispatch('pointerup');
		expect(handlers.onCommit).not.toHaveBeenCalled();
	});

	it('reports lane fractions while moving and commits on pointer up', () => {
		const handlers = { onMove: vi.fn(), onCommit: vi.fn(), onCancel: vi.fn() };
		expect(beginLaneDrag(pointerDown(), lane(0, 100), handlers)).toBe(true);
		dispatch('pointermove', { clientX: 25 });
		dispatch('pointermove', { clientX: 400 });
		expect(handlers.onMove.mock.calls).toEqual([[0.25], [1]]);
		dispatch('pointerup');
		expect(handlers.onCommit).toHaveBeenCalledTimes(1);
		expect(handlers.onCancel).not.toHaveBeenCalled();
		dispatch('pointermove', { clientX: 50 });
		expect(handlers.onMove).toHaveBeenCalledTimes(2);
	});

	it('cancels on Escape', () => {
		const handlers = { onMove: vi.fn(), onCommit: vi.fn(), onCancel: vi.fn() };
		beginLaneDrag(pointerDown(), lane(0, 100), handlers);
		dispatch('keydown', { key: 'Escape' });
		expect(handlers.onCancel).toHaveBeenCalledTimes(1);
		expect(handlers.onCommit).not.toHaveBeenCalled();
	});
});
