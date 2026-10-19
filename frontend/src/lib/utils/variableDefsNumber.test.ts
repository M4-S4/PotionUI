import { describe, expect, it } from 'vitest';
import {
	MAX_NUMBER_GRID,
	buildNumberWireValue,
	buildVariablesForSubmit,
	clampNumberValue,
	createNumberVariable,
	isVariableLinked,
	numberChipLabel,
	numberGridSize,
	numberGridValue,
	numberGridValues,
	numberRangeLabel,
	rollNumberValue,
	seededRandom,
	usagesShareRoll,
	type NumberVariableDef,
	type VariablesMap
} from './variableDefs';
import { resolvePreviewVariables } from './resolvedVariables';

const age = (over: Partial<NumberVariableDef> = {}): NumberVariableDef => ({
	type: 'number',
	mode: 'shuffle',
	min: 18,
	max: 80,
	step: 1,
	value: 47,
	...over
});

describe('number grid', () => {
	it('counts min + k*step values inclusive', () => {
		expect(numberGridSize(age())).toBe(63);
		expect(numberGridSize(age({ step: 5 }))).toBe(13);
		expect(numberGridSize(age({ min: 10, max: 10 }))).toBe(1);
	});

	it('is empty for an invalid range', () => {
		expect(numberGridSize(age({ step: 0 }))).toBe(0);
		expect(numberGridSize(age({ min: 5, max: 1 }))).toBe(0);
		expect(numberGridSize(age({ min: NaN }))).toBe(0);
	});

	it('formats decimals from the step without float drift', () => {
		const weight = age({ min: 0.8, max: 1.4, step: 0.1 });
		expect(numberGridValues(weight)).toEqual(['0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4']);
		expect(numberGridValue(age({ min: 0, max: 1, step: 0.05 }), 3)).toBe('0.15');
		expect(numberGridValue(age({ min: 0.1, max: 1, step: 0.1 }), 2)).toBe('0.3');
	});

	it('steps an integer grid by the step', () => {
		expect(numberGridValues(age({ step: 5, max: 38 }))).toEqual(['18', '23', '28', '33', '38']);
	});

	it('limits the label to the grid decimals', () => {
		expect(numberRangeLabel(age({ min: 0.8, max: 1.4, step: 0.1 }))).toBe('0.8–1.4');
		expect(numberRangeLabel(age())).toBe('18–80');
	});
});

describe('clamp', () => {
	it('keeps a pinned value inside the range', () => {
		expect(clampNumberValue(age({ value: 99 }))).toBe('80');
		expect(clampNumberValue(age({ value: 3 }))).toBe('18');
		expect(clampNumberValue(age({ value: 47 }))).toBe('47');
	});

	it('formats with the step decimals', () => {
		expect(clampNumberValue(age({ min: 0, max: 2, step: 0.1, value: 1.25 }))).toBe('1.3');
		expect(clampNumberValue(age({ min: 0, max: 2, step: 0.1, value: 3 }))).toBe('2.0');
	});
});

describe('roll', () => {
	it('picks uniformly over the grid from the injected source', () => {
		const def = age({ step: 5 });
		expect(rollNumberValue(def, () => 0)).toEqual({ index: 0, value: '18' });
		expect(rollNumberValue(def, () => 0.999999)).toEqual({ index: 12, value: '78' });
		expect(rollNumberValue(def, () => 0.5)).toEqual({ index: 6, value: '48' });
	});

	it('is deterministic for a seed and varies across seeds', () => {
		const roll = (seed: number) => {
			const random = seededRandom(seed);
			return Array.from({ length: 5 }, () => rollNumberValue(age(), random)?.value);
		};
		expect(roll(7)).toEqual(roll(7));
		expect(roll(7)).not.toEqual(roll(8));
	});

	it('does not roll an invalid or oversized grid', () => {
		expect(rollNumberValue(age({ step: 0 }))).toBeNull();
		expect(rollNumberValue(age({ min: 0, max: 1000 }))).toBeNull();
	});
});

describe('wire serialization', () => {
	it('pin sends the clamped literal', () => {
		expect(buildNumberWireValue(age({ mode: 'pin', value: 47 }))).toBe('47');
		expect(buildNumberWireValue(age({ mode: 'pin', value: 200 }))).toBe('80');
	});

	it('per image sends the enumerated group', () => {
		expect(buildNumberWireValue(age({ mode: 'per-image', min: 18, max: 21 }))).toBe('{18|19|20|21}');
		expect(buildNumberWireValue(age({ mode: 'per-image', min: 0.8, max: 1, step: 0.1 }))).toBe('{0.8|0.9|1.0}');
	});

	it('a one-value grid is the literal', () => {
		expect(buildNumberWireValue(age({ mode: 'per-image', min: 5, max: 5 }))).toBe('5');
	});

	it('caps the grid at 200 values', () => {
		const exact = age({ mode: 'per-image', min: 1, max: MAX_NUMBER_GRID });
		expect(buildNumberWireValue(exact)).toMatch(/^\{1\|2\|/);
		expect(buildNumberWireValue({ ...exact, max: MAX_NUMBER_GRID + 1 })).toBe('');
	});

	it('sends nothing for an invalid range', () => {
		expect(buildNumberWireValue(age({ mode: 'pin', step: 0 }))).toBe('');
	});
});

describe('buildVariablesForSubmit with numbers', () => {
	it('shuffle rolls once and shares the pick', () => {
		const vars: VariablesMap = { age: age({ step: 5 }) };
		const { wireMap, rolls } = buildVariablesForSubmit(vars, { random: () => 0.5, now: () => 1 });
		expect(wireMap).toEqual({ age: '48' });
		expect(rolls.age).toEqual({ optionIndex: 6, value: '48', rolledAt: 1 });
	});

	it('independent usages roll separately', () => {
		const vars: VariablesMap = { age: age({ step: 5 }) };
		const sequence = [0, 0.5, 0.999999];
		let i = 0;
		const { wireMap } = buildVariablesForSubmit(vars, {
			random: () => sequence[i++],
			usageCounts: { age: 3 }
		});
		expect(wireMap).toEqual({ age: '18', 'age-2': '48', 'age-3': '78' });
	});

	it('pin and per image are literal and group', () => {
		const pinned = buildVariablesForSubmit({ age: age({ mode: 'pin', value: 47 }) });
		expect(pinned.wireMap).toEqual({ age: '47' });
		expect(pinned.rolls).toEqual({});
		const per = buildVariablesForSubmit({ age: age({ mode: 'per-image', min: 1, max: 3 }) });
		expect(per.wireMap).toEqual({ age: '{1|2|3}' });
	});

	it('omits an unrollable number', () => {
		expect(buildVariablesForSubmit({ age: age({ step: 0 }) }).wireMap).toEqual({});
	});

	it('is deterministic with a seeded source', () => {
		const run = () => buildVariablesForSubmit({ age: age() }, { random: seededRandom(42), now: () => 0 });
		expect(run()).toEqual(run());
	});
});

describe('linking', () => {
	it('shuffle numbers are independent unless flagged or read by a condition', () => {
		expect(isVariableLinked('age', { age: age() })).toBe(false);
		expect(isVariableLinked('age', { age: age({ linked: true }) })).toBe(true);
		const readers = { age: [{ label: 'age > 65' }] };
		expect(isVariableLinked('age', { age: age() }, readers)).toBe(true);
		expect(isVariableLinked('age', { age: age() }, { other: [{ label: 'x' }] })).toBe(false);
	});

	it('pin and per image are linked, but only per image reads share a roll through a condition', () => {
		expect(isVariableLinked('age', { age: age({ mode: 'pin' }) })).toBe(true);
		expect(usagesShareRoll('age', { age: age({ mode: 'per-image' }) })).toBe(false);
		const readers = { age: [{ label: 'age > 65' }] };
		expect(usagesShareRoll('age', { age: age({ mode: 'per-image' }) }, readers)).toBe(true);
	});
});

describe('labels and previews', () => {
	it('chip label follows the mode', () => {
		expect(numberChipLabel(age())).toBe('18–80');
		expect(numberChipLabel(age({ mode: 'pin' }))).toBe('47');
		expect(numberChipLabel(age({ mode: 'per-image' }))).toBe('per image');
	});

	it('resolved preview notes follow the same wording as choice', () => {
		const shuffle = resolvePreviewVariables({ age: age() });
		expect(shuffle.notes).toEqual([{ name: 'age', phrase: 'shuffles each generation' }]);
		expect(shuffle.replacements.age).toBe('18');
		const pinned = resolvePreviewVariables({ age: age({ mode: 'pin', value: 47 }) });
		expect(pinned.notes).toEqual([]);
		expect(pinned.replacements.age).toBe('47');
		const per = resolvePreviewVariables({ age: age({ mode: 'per-image' }) });
		expect(per.notes).toEqual([{ name: 'age', phrase: 're-rolls per image' }]);
		expect(per.replacements.age).toBeUndefined();
	});

	it('createNumberVariable is a valid range', () => {
		expect(numberGridSize(createNumberVariable())).toBe(101);
	});
});
