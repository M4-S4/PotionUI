import { describe, expect, it, vi } from 'vitest';
import {
	canReturnToDetails,
	rememberDetailsScroll,
	registerDetailsReturn,
	returnToDetails,
	takeDetailsScroll
} from './detailsReturn';

describe('detailsReturn', () => {
	it('does nothing and says so when no History page is listening', () => {
		expect(canReturnToDetails()).toBe(false);
		expect(returnToDetails({ generationId: 'g', fileIndex: 0 })).toBe(false);
	});

	it('hands the generation and file index to the listening page', () => {
		const handler = vi.fn();
		const unregister = registerDetailsReturn(handler);

		expect(canReturnToDetails()).toBe(true);
		expect(returnToDetails({ generationId: 'g1', fileIndex: 3 })).toBe(true);
		expect(handler).toHaveBeenCalledWith({ generationId: 'g1', fileIndex: 3 });
		unregister();
		expect(canReturnToDetails()).toBe(false);
	});

	it('keeps a newer listener when an older page unregisters late', () => {
		const first = vi.fn();
		const second = vi.fn();
		const unregisterFirst = registerDetailsReturn(first);
		registerDetailsReturn(second);

		unregisterFirst();
		returnToDetails({ generationId: 'g', fileIndex: 0 });

		expect(second).toHaveBeenCalledTimes(1);
		expect(first).not.toHaveBeenCalled();
	});
});

describe('details scroll memory', () => {
	function comeBack(id: string) {
		const unregister = registerDetailsReturn(() => {});
		returnToDetails({ generationId: id, fileIndex: 0 });
		unregister();
	}

	it('hands the remembered scroll back once, when the page returns to that generation', () => {
		rememberDetailsScroll('g1', 320);
		comeBack('g1');

		expect(takeDetailsScroll('g2')).toBeNull();
		expect(takeDetailsScroll('g1')).toBe(320);
		expect(takeDetailsScroll('g1')).toBeNull();
	});

	it('does not restore when the generation is opened afresh rather than returned to', () => {
		rememberDetailsScroll('g1', 320);

		expect(takeDetailsScroll('g1')).toBeNull();
	});

	it('keeps nothing for a window that was not scrolled', () => {
		rememberDetailsScroll('g1', 320);
		rememberDetailsScroll('g1', 0);
		comeBack('g1');

		expect(takeDetailsScroll('g1')).toBeNull();
	});
});
