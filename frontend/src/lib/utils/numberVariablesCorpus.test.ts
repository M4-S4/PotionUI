import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
	buildNumberWireValue,
	isNumberRangeValid,
	numberGridSize,
	numberGridValue,
	type NumberVariableDef
} from './variableDefs';

interface CorpusCase {
	name: string;
	def: Record<string, unknown>;
	expected: { valid: boolean; count: number; first: string; last: string; wire: string };
}

const corpusPath = resolve(__dirname, '../../../../tests/fixtures/prompt_conditions/number_variables.json');
const corpus = JSON.parse(readFileSync(corpusPath, 'utf8')) as { cases: CorpusCase[] };

describe('number variable corpus parity', () => {
	it('loads the full corpus', () => {
		expect(corpus.cases.length).toBe(38);
	});

	it.each(corpus.cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
		const def = { type: 'number', ...c.def } as unknown as NumberVariableDef;
		expect(isNumberRangeValid(def)).toBe(c.expected.valid);
		expect(buildNumberWireValue(def)).toBe(c.expected.wire);
		expect(c.expected.valid ? numberGridSize(def) : 0).toBe(c.expected.count);
		if (c.expected.valid) {
			expect(numberGridValue(def, 0)).toBe(c.expected.first);
			expect(numberGridValue(def, c.expected.count - 1)).toBe(c.expected.last);
		}
	});
});
