import { describe, it, expect } from 'vitest';
import { jsonEqual } from './equal';

describe('jsonEqual', () => {
	it('compares structure, not identity', () => {
		expect(jsonEqual({ a: [1, 2] }, { a: [1, 2] })).toBe(true);
		expect(jsonEqual({ a: [1, 2] }, { a: [2, 1] })).toBe(false);
	});

	it('is key-order sensitive', () => {
		expect(jsonEqual({ a: 1, b: 2 }, { b: 2, a: 1 })).toBe(false);
	});

	it('does not treat blank values as equal', () => {
		expect(jsonEqual(undefined, null)).toBe(false);
		expect(jsonEqual('', null)).toBe(false);
		expect(jsonEqual(undefined, undefined)).toBe(true);
	});
});
