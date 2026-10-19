import { describe, expect, it } from 'vitest';
import {
	conditionIssue,
	conditionReadersFor,
	conditionTagWarnings,
	conditionTooltip,
	conditionValues,
	countConditionBlocks,
	defaultConditionRaw,
	draftFromRaw,
	lastRunFor,
	operatorsFor,
	plainConditionSummary,
	rawFromDraft,
	retargetTest,
	setTestOperator,
	summarizeCondition,
	testFromDraft
} from './conditionTags';
import type { VariablesMap } from './variableDefs';

const variables: VariablesMap = {
	age: { type: 'number', mode: 'per-image', min: 18, max: 80, step: 1, value: 18 },
	mood: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['calm', 'serene', 'tense'] },
	name: { type: 'text', value: 'Mara' }
};

const CANONICAL =
	'<<if $age > 65>>grey hair, deep wrinkles<<else if $age < 25 and $mood in "calm", "serene">>youthful, soft skin<<else>>auburn hair<<end>>';

describe('draft round trip', () => {
	it('parses the canonical block into branches and writes it back unchanged', () => {
		const draft = draftFromRaw(CANONICAL)!;
		expect(draft.branches.map((b) => b.kind)).toEqual(['if', 'elseif', 'else']);
		expect(draft.branches[1].tests.map((t) => t.var)).toEqual(['age', 'mood']);
		expect(draft.branches[1].tests[1].strings).toEqual(['calm', 'serene']);
		expect(rawFromDraft(draft, variables)).toBe(CANONICAL);
	});

	it('refuses text that is not exactly one well-formed block', () => {
		expect(draftFromRaw('a <<if $age > 1>>x<<end>>')).toBeNull();
		expect(draftFromRaw('<<if $age > 1>>x')).toBeNull();
		expect(draftFromRaw('<<if $age > 1>>x<<end>> y')).toBeNull();
	});

	it('does not write an incomplete or nested edit', () => {
		const draft = draftFromRaw(CANONICAL)!;
		draft.branches[0].tests[0].number = '6.';
		expect(rawFromDraft(draft, variables)).toBeNull();
		const nested = draftFromRaw(CANONICAL)!;
		nested.branches[0].body = 'a <<if $age > 1>>b<<end>>';
		expect(rawFromDraft(nested, variables)).toBeNull();
	});

	it('quotes and escapes text operands', () => {
		const draft = draftFromRaw('<<if $name = "x">>a<<end>>')!;
		draft.branches[0].tests[0].strings = ['say "hi"'];
		expect(rawFromDraft(draft, variables)).toBe('<<if $name = "say \\"hi\\"">>a<<end>>');
	});

	it('limits tests to six and strings to 80 characters', () => {
		const draft = draftFromRaw('<<if $name = "x">>a<<end>>')!;
		draft.branches[0].tests[0].strings = ['x'.repeat(81)];
		expect(rawFromDraft(draft, variables)).toBeNull();
		const six = draftFromRaw('<<if $name = "x">>a<<end>>')!;
		six.branches[0].tests = Array.from({ length: 7 }, () => ({ ...six.branches[0].tests[0] }));
		expect(rawFromDraft(six, variables)).toBeNull();
	});
});

describe('operators and operands by variable type', () => {
	it('offers the operators from the grammar table', () => {
		expect(operatorsFor('number').map((o) => o.op)).toEqual(['=', '!=', '<', '<=', '>', '>=', 'in', 'not in']);
		expect(operatorsFor('choice').map((o) => o.op)).toEqual(['=', '!=', 'in', 'not in', 'is empty', 'is not empty']);
		expect(operatorsFor('text').map((o) => o.op)).toEqual(['=', '!=', 'contains', 'not contains', 'is empty', 'is not empty']);
	});

	it('writes a between test as a range and rejects a reversed one', () => {
		const draft = draftFromRaw('<<if $age > 1>>a<<end>>')!;
		const test = setTestOperator(draft.branches[0].tests[0], 'in', variables);
		expect(testFromDraft({ ...test, low: '18', high: '30' }, variables)).toMatchObject({ kind: 'range', low: '18', high: '30' });
		expect(testFromDraft({ ...test, low: '30', high: '18' }, variables)).toBeNull();
	});

	it('retargets to a variable of another type with that type\'s defaults', () => {
		const draft = draftFromRaw('<<if $age > 65>>a<<end>>')!;
		const moved = retargetTest(draft.branches[0].tests[0], 'mood', variables);
		expect(moved).toMatchObject({ var: 'mood', op: '=', strings: ['calm'] });
		const same = retargetTest(draft.branches[0].tests[0], 'age', variables);
		expect(same.op).toBe('>');
	});
});

describe('defaultConditionRaw', () => {
	it('numbers start at the range minimum', () => {
		expect(defaultConditionRaw({ age: variables.age })).toBe('<<if $age >= 18>>text<<end>>');
	});

	it('choices start on their first option', () => {
		expect(defaultConditionRaw({ mood: variables.mood }, 'x')).toBe('<<if $mood = "calm">>x<<end>>');
	});

	it('text variables start with is empty and no variables use a placeholder name', () => {
		expect(defaultConditionRaw({ name: variables.name })).toBe('<<if $name is empty>>text<<end>>');
		expect(defaultConditionRaw({})).toBe('<<if $value is empty>>text<<end>>');
	});
});

describe('warnings', () => {
	it('flags a deleted variable', () => {
		expect(conditionIssue('<<if $ghost > 1>>a<<end>>', variables)).toBe("$ghost isn't a variable on this tab.");
	});

	it('flags a type change', () => {
		expect(conditionIssue('<<if $age = "senior">>a<<end>>', variables)).toContain('age is a number now');
		expect(conditionIssue('<<if $mood > 3>>a<<end>>', variables)).toContain('mood is a choice variable now');
	});

	it('flags a number that can never match', () => {
		expect(conditionIssue('<<if $age > 90>>a<<end>>', variables)).toContain('only goes between 18 and 80');
		expect(conditionIssue('<<if $age > 65>>a<<end>>', variables)).toBeNull();
		expect(conditionIssue('<<if $age in 90..99>>a<<end>>', variables)).not.toBeNull();
	});

	it('flags a choice option that is gone', () => {
		expect(conditionIssue('<<if $mood = "angry">>a<<end>>', variables)).toBe('angry is no longer an option of $mood.');
		expect(conditionIssue('<<if $mood in "calm", "angry">>a<<end>>', variables)).toBe('angry is no longer an option of $mood.');
		expect(conditionIssue('<<if $mood = "calm">>a<<end>>', variables)).toBeNull();
	});

	it('marks hand-typed grammar errors for highlighting', () => {
		expect(conditionTagWarnings('woman <<if $age > 65>>grey hair')).toEqual([{ start: 6, end: 22, tone: 'warning' }]);
		expect(conditionTagWarnings('fine <<if $age > 65>>x<<end>>')).toEqual([]);
		expect(conditionTagWarnings('no tags')).toEqual([]);
	});
});

describe('summary', () => {
	it('summarizes the first test, its text, the otherwise and the hidden branches', () => {
		expect(summarizeCondition(CANONICAL)).toEqual({
			head: 'age > 65',
			then: 'grey hair, deep wrinkles',
			otherwise: 'auburn hair',
			extra: 1
		});
		expect(summarizeCondition('<<if $age > 65>>grey<<end>>')).toMatchObject({ otherwise: null, extra: 0 });
		expect(summarizeCondition('not a block')).toBeNull();
	});

	it('writes the full condition into the tooltip', () => {
		expect(conditionTooltip(CANONICAL).split('\n')).toEqual([
			'if age > 65 → grey hair, deep wrinkles',
			'else if age < 25 and mood in "calm", "serene" → youthful, soft skin',
			'otherwise → auburn hair'
		]);
	});

	it('counts blocks', () => {
		expect(countConditionBlocks(`a ${CANONICAL} b ${CANONICAL}`)).toBe(2);
		expect(countConditionBlocks('plain')).toBe(0);
		expect(countConditionBlocks('<<if $a > 1>>x')).toBe(0);
	});
});

describe('readers', () => {
	it('maps each read variable to the segments and heads that read it', () => {
		const readers = conditionReadersFor(['plain', `a ${CANONICAL}`]);
		expect(Object.keys(readers).sort()).toEqual(['age', 'mood']);
		expect(readers.age).toEqual([
			{
				label: 'age > 65',
				segment: 'segment 02',
				splits: [{ from: 65, to: Infinity, label: 'IF > 65', align: 'end' }]
			}
		]);
		expect(readers.mood[0].label).toBe('age < 25 and mood in "calm", "serene"');
	});

	it('records every number split for the roll track', () => {
		const readers = conditionReadersFor(['<<if $age > 65>>a<<else if $age < 25>>b<<end>>']);
		expect(readers.age).toHaveLength(1);
		expect(readers.age[0].splits).toEqual([{ from: 65, to: Infinity, label: 'IF > 65', align: 'end' }]);
	});

	it('is empty when no segment has a condition', () => {
		expect(conditionReadersFor(['a ${age} b', ''])).toEqual({});
	});
});

describe('values and last run', () => {
	it('uses text, pins and the last roll but not per-image variables', () => {
		const vars: VariablesMap = {
			...variables,
			mood: { type: 'choice', mode: 'pin', pinnedIndex: 1, options: ['calm', 'serene'] },
			tone: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['a', 'b'] },
			weight: { type: 'number', mode: 'shuffle', min: 0, max: 5, step: 1, value: 0 },
			level: { type: 'number', mode: 'pin', min: 0, max: 5, step: 1, value: 9 }
		};
		const rolls = {
			tone: { optionIndex: 1, value: 'b', rolledAt: 1 },
			weight: { optionIndex: 3, value: '3', rolledAt: 1 }
		};
		expect(conditionValues(vars, rolls)).toEqual({ name: 'Mara', mood: 'serene', tone: 'b', weight: '3', level: '5' });
	});

	it('names the branch a settled run takes and waits otherwise', () => {
		const settled = lastRunFor('<<if $mood = "calm">>x<<else>>y<<end>>', { mood: 'calm' });
		expect(settled).toMatchObject({ settled: true, position: 0, body: 'x', reads: [{ name: 'mood', value: 'calm' }] });
		const waiting = lastRunFor('<<if $mood = "calm">>x<<else>>y<<end>>', {});
		expect(waiting).toMatchObject({ settled: false, reads: [{ name: 'mood', value: null }] });
	});
});

describe('plain summary', () => {
	it('words the first test in plain language', () => {
		expect(plainConditionSummary('<<if $age >= 0>>text<<end>>')).toBe('When age is at least 0');
		expect(plainConditionSummary('<<if $age in 18..30>>a<<end>>')).toBe('When age is between 18 and 30');
		expect(plainConditionSummary('<<if $mood in "calm", "serene">>a<<end>>')).toBe('When mood is one of calm, serene');
		expect(plainConditionSummary('<<if $name contains "Ma">>a<<end>>')).toBe('When name contains Ma');
		expect(plainConditionSummary('<<if $name is empty>>a<<end>>')).toBe('When name is empty');
	});

	it('joins several tests and counts the other branches', () => {
		expect(plainConditionSummary(CANONICAL)).toBe('When age is more than 65 +2 more');
		expect(plainConditionSummary('<<if $age > 1 or $age < 0>>a<<else>>b<<end>>')).toBe(
			'When age is more than 1 or age is less than 0 +1 more'
		);
	});

	it('falls back for a malformed block', () => {
		expect(plainConditionSummary('<<if $age >>oops<<end>>')).toBe('Condition');
	});
});
