import { describe, it, expect } from 'vitest';
import { stepIndex } from './listNavigation';

describe('stepIndex', () => {
	it('moves forward and wraps from the last item to the first', () => {
		expect(stepIndex(0, 1, 3)).toBe(1);
		expect(stepIndex(2, 1, 3)).toBe(0);
	});

	it('moves back and wraps from the first item to the last', () => {
		expect(stepIndex(2, -1, 3)).toBe(1);
		expect(stepIndex(0, -1, 3)).toBe(2);
	});

	it('stays on the only item of a one-item list', () => {
		expect(stepIndex(0, 1, 1)).toBe(0);
		expect(stepIndex(0, -1, 1)).toBe(0);
	});

	it('stays at 0 for an empty list in both directions, never NaN or -1', () => {
		expect(stepIndex(0, 1, 0)).toBe(0);
		expect(stepIndex(0, -1, 0)).toBe(0);
		expect(stepIndex(5, 1, 0)).toBe(0);
	});

	it('brings an index left past the end by a shrunk list back inside it', () => {
		expect(stepIndex(7, 1, 3)).toBe(2);
		expect(stepIndex(7, -1, 3)).toBe(0);
	});
});
