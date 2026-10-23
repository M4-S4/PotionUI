import { describe, it, expect } from 'vitest';
import { applyVariableEdits } from './variableEdits';

const text = (value: string) => ({ type: 'text' as const, value });

describe('applyVariableEdits', () => {
	it('applies only the changed, added and removed names on top of the current map', () => {
		const from = { a: text('1'), b: text('2'), c: text('3') };
		const to = { a: text('1'), b: text('two'), d: text('4') };
		const current = { a: text('1'), b: text('2'), c: text('3'), pushed: text('p') };
		expect(applyVariableEdits(current, from, to)).toEqual({
			a: text('1'),
			b: text('two'),
			d: text('4'),
			pushed: text('p')
		});
	});

	it('returns an equal map when nothing was edited', () => {
		const from = { a: text('1') };
		expect(applyVariableEdits({ a: text('1'), x: text('2') }, from, { a: text('1') })).toEqual({
			a: text('1'),
			x: text('2')
		});
	});

	it('treats undefined-valued keys as absent', () => {
		const from = { a: { type: 'text' as const, value: '1', linked: undefined } };
		expect(applyVariableEdits({ a: text('9') }, from, { a: text('1') })).toEqual({ a: text('9') });
	});
});
