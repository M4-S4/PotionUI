import { describe, expect, it } from 'vitest';
import {
	branchName,
	branchProblemText,
	catchingBranch,
	checkDraft,
	choiceOptions,
	conditionIssue,
	conditionReadersFor,
	conditionTagWarnings,
	conditionTooltip,
	conditionValues,
	countConditionBlocks,
	defaultConditionRaw,
	defaultTest,
	draftFromRaw,
	draftProblemText,
	firstUncoveredOption,
	fitsCondition,
	lastRunFor,
	nestedToElseIf,
	nextBranchTest,
	operatorsFor,
	optionUsage,
	plainConditionSummary,
	rawFromDraft,
	retargetTest,
	setTestOperator,
	summarizeCondition,
	testFromDraft,
	type ConditionDraft
} from './conditionTags';
import { MAX_STRING_CHARS, MAX_TESTS } from './promptConditions';
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

	it('limits tests to six and strings to the value cap', () => {
		const draft = draftFromRaw('<<if $name = "x">>a<<end>>')!;
		draft.branches[0].tests[0].strings = ['x'.repeat(MAX_STRING_CHARS + 1)];
		expect(rawFromDraft(draft, variables)).toBeNull();
		const six = draftFromRaw('<<if $name = "x">>a<<end>>')!;
		six.branches[0].tests = Array.from({ length: 7 }, () => ({ ...six.branches[0].tests[0] }));
		expect(rawFromDraft(six, variables)).toBeNull();
	});
});

describe('draft problems', () => {
	const draftOf = (raw: string): ConditionDraft => draftFromRaw(raw)!;

	it('writes a complete draft with no problems', () => {
		expect(checkDraft(draftOf(CANONICAL), variables)).toEqual({ raw: CANONICAL, problems: [] });
	});

	it('names the branch, the test and the reason of each unfinished value', () => {
		const draft = draftOf('<<if $name = "a">>x<<else if $age in 18..30 and $mood in "calm">>y<<end>>');
		draft.branches[0].tests[0].strings = [''];
		draft.branches[1].tests[0].low = '40';
		draft.branches[1].tests[1].strings = Array.from({ length: 13 }, (_, i) => `v${i}`);
		expect(checkDraft(draft, variables)).toEqual({
			raw: null,
			problems: [
				{ kind: 'value', branch: 0, test: 0, reason: 'empty' },
				{ kind: 'value', branch: 1, test: 0, reason: 'reversed' },
				{ kind: 'value', branch: 1, test: 1, reason: 'too-many' }
			]
		});
	});

	it('tells a bad number and a value over the limit apart', () => {
		const draft = draftOf('<<if $age > 1 and $name contains "a">>x<<end>>');
		draft.branches[0].tests[0].number = '6.';
		draft.branches[0].tests[1].strings = ['x'.repeat(MAX_STRING_CHARS + 1)];
		expect(checkDraft(draft, variables).problems).toEqual([
			{ kind: 'value', branch: 0, test: 0, reason: 'number' },
			{ kind: 'value', branch: 0, test: 1, reason: 'too-long' }
		]);
	});

	it('counts the length limit in characters, like the parser', () => {
		const draft = draftOf('<<if $name = "a">>x<<end>>');
		draft.branches[0].tests[0].strings = ['🙂'.repeat(MAX_STRING_CHARS)];
		const raw = rawFromDraft(draft, variables);
		expect(raw).not.toBeNull();
		expect(draftFromRaw(raw!)).not.toBeNull();
	});

	it('flags nested text, a branch without tests and too many branches', () => {
		const draft = draftOf('<<if $age > 1>>x<<else>>y<<end>>');
		draft.branches[1].body = 'y <<if $age > 2>>z<<end>>';
		draft.branches[0].tests = [];
		expect(checkDraft(draft, variables).problems).toEqual([
			{ kind: 'tests', branch: 0 },
			{ kind: 'nested', branch: 1 }
		]);
		const many = draftOf('<<if $age > 1>>x<<end>>');
		many.branches = Array.from({ length: 9 }, (_, i) => ({ ...many.branches[0], kind: i === 0 ? 'if' : 'elseif' }));
		expect(checkDraft(many, variables).problems).toEqual([{ kind: 'branches' }]);
	});

	it('refuses a head the parser would not read back', () => {
		const long = draftOf('<<if $name = "a">>x<<end>>');
		long.branches[0].tests = Array.from({ length: MAX_TESTS }, () => ({
			...long.branches[0].tests[0],
			strings: ['x'.repeat(MAX_STRING_CHARS)]
		}));
		expect(checkDraft(long, variables).problems).toEqual([{ kind: 'head', branch: 0 }]);
		const name = 'v'.repeat(61);
		const unreadable = draftOf('<<if $age is empty>>x<<end>>');
		unreadable.branches[0].tests[0].var = name;
		expect(checkDraft(unreadable, variables)).toEqual({ raw: null, problems: [{ kind: 'grammar' }] });
	});
});

describe('problem messages', () => {
	const draft = draftFromRaw(
		'<<if $mood = "calm">>a<<else if $mood in "calm">>b<<else if $age in 18..30>>c<<else>>d<<end>>'
	)!;

	it('names branches the way the editor shows them', () => {
		expect([0, 1, 2, 3].map((i) => branchName(draft, i))).toEqual(['If', 'Else if 1', 'Else if 2', 'Otherwise']);
		const single = draftFromRaw('<<if $age > 1>>a<<else if $age > 2>>b<<end>>')!;
		expect(branchName(single, 1)).toBe('Else if');
	});

	it('words each value reason for the variable it is about', () => {
		const text = (branch: number, reason: 'empty' | 'too-long' | 'number' | 'reversed' | 'too-many') =>
			draftProblemText({ kind: 'value', branch, test: 0, reason }, draft, variables);
		expect(text(0, 'empty')).toBe('If: pick a value for $mood.');
		expect(text(1, 'empty')).toBe('Else if 1: pick at least one value for $mood.');
		expect(text(0, 'too-long')).toBe(
			`If: the option picked for $mood is longer than ${MAX_STRING_CHARS} characters and can't be used in a condition.`
		);
		expect(text(1, 'too-many')).toBe('Else if 1: pick at most 12 values for $mood.');
		expect(text(2, 'reversed')).toBe('Else if 2: the From number for $age is above To.');
		expect(text(2, 'number')).toBe('Else if 2: enter a number for $age.');
		const named = draftFromRaw('<<if $name = "a">>a<<end>>')!;
		expect(draftProblemText({ kind: 'value', branch: 0, test: 0, reason: 'empty' }, named, variables)).toBe(
			'If: type the text to compare $name with.'
		);
	});

	it('words branch-wide problems', () => {
		expect(draftProblemText({ kind: 'nested', branch: 3 }, draft, variables)).toBe(
			"Otherwise text contains another condition. Conditions can't be nested, so add an Else if instead."
		);
		expect(draftProblemText({ kind: 'tests', branch: 1 }, draft, variables)).toBe('Else if 1 needs 1 to 6 tests.');
		expect(draftProblemText({ kind: 'head', branch: 0 }, draft, variables)).toBe(
			'If is too long to save. Remove a test or pick shorter values.'
		);
		expect(draftProblemText({ kind: 'branches' }, draft, variables)).toBe('A condition can have at most 8 branches.');
		expect(draftProblemText({ kind: 'grammar' }, draft, variables)).toContain("can't be saved as it is");
	});
});

describe('nested condition to else if', () => {
	const withOtherwise = (body: string) => {
		const draft = draftFromRaw('<<if $age > 65>>grey<<else>>x<<end>>')!;
		draft.branches[1].body = body;
		return draft;
	};

	it('turns an otherwise that is only a condition into its branches', () => {
		const next = nestedToElseIf(withOtherwise(' <<if $age < 25>>young<<else if $age < 40>>mid<<else>>auburn<<end>> '))!;
		expect(rawFromDraft(next, variables)).toBe(
			'<<if $age > 65>>grey<<else if $age < 25>>young<<else if $age < 40>>mid<<else>>auburn<<end>>'
		);
	});

	it('leaves shapes whose meaning would change', () => {
		expect(nestedToElseIf(withOtherwise('auburn <<if $age < 25>>young<<end>>'))).toBeNull();
		expect(nestedToElseIf(withOtherwise('<<if $age < 25>>young<<end>> auburn'))).toBeNull();
		const inIf = draftFromRaw('<<if $age > 65>>grey<<end>>')!;
		inIf.branches[0].body = '<<if $age < 70>>young<<end>>';
		expect(nestedToElseIf(inIf)).toBeNull();
	});

	it('stops at the branch limit', () => {
		const heads = Array.from({ length: 6 }, (_, i) => `<<else if $age = ${i + 20}>>b${i}`).join('');
		const outer = draftFromRaw(`<<if $age > 65>>a${heads}<<else>>x<<end>>`)!;
		outer.branches[outer.branches.length - 1].body = '<<if $age < 5>>c<<else>>d<<end>>';
		expect(outer.branches.length).toBe(8);
		expect(nestedToElseIf(outer)).toBeNull();
	});
});

describe('default values', () => {
	const long = 'x'.repeat(MAX_STRING_CHARS + 1);
	const looks: VariablesMap = {
		look: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: [long, 'linen', 'silk'] },
		name: variables.name
	};

	it('starts a choice test on the first option short enough to compare', () => {
		expect(defaultTest('look', looks)).toMatchObject({ op: '=', strings: ['linen'] });
		expect(fitsCondition('x'.repeat(MAX_STRING_CHARS))).toBe(true);
		expect(fitsCondition(long)).toBe(false);
	});

	it('fills a choice value when a test leaves is empty, but leaves text for the user', () => {
		const empty = draftFromRaw('<<if $look is empty>>a<<end>>')!.branches[0].tests[0];
		expect(setTestOperator(empty, '=', looks).strings).toEqual(['linen']);
		const text = draftFromRaw('<<if $name is empty>>a<<end>>')!.branches[0].tests[0];
		expect(setTestOperator(text, '=', looks).strings).toEqual(['']);
	});

	it('keeps one value when a list test becomes a single comparison', () => {
		const listed = draftFromRaw('<<if $look in "linen", "silk">>a<<end>>')!.branches[0].tests[0];
		expect(setTestOperator(listed, '=', looks).strings).toEqual(['linen']);
		expect(setTestOperator(listed, 'not in', looks).strings).toEqual(['linen', 'silk']);
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

describe('options across branches', () => {
	const vars: VariablesMap = {
		...variables,
		look: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: [' suit ', 'gown', 'suit', '', 'jeans', 'x'.repeat(MAX_STRING_CHARS + 1)] }
	};
	const SWITCH = '<<if $look = "suit">>a<<else if $look in "gown", "jeans">>b<<else if $look != "gown">>c<<else>>d<<end>>';

	it('lists a choice variable’s options once, trimmed', () => {
		expect(choiceOptions('look', vars)).toEqual(['suit', 'gown', 'jeans', 'x'.repeat(MAX_STRING_CHARS + 1)]);
		expect(choiceOptions('age', vars)).toEqual([]);
		expect(choiceOptions('missing', vars)).toEqual([]);
	});

	it('maps each option to the other branches that test it, with the operator', () => {
		const draft = draftFromRaw(SWITCH)!;
		const usage = optionUsage(draft, 0, 'look');
		expect(usage.get('gown')).toEqual([
			{ branch: 1, op: 'in' },
			{ branch: 2, op: '!=' }
		]);
		expect(usage.get('jeans')).toEqual([{ branch: 1, op: 'in' }]);
		expect(usage.has('suit')).toBe(false);
		expect(optionUsage(draft, 1, 'look').get('suit')).toEqual([{ branch: 0, op: '=' }]);
		expect(optionUsage(draft, 1, 'mood').size).toBe(0);
	});

	it('finds the earlier branch that already catches a value', () => {
		const draft = draftFromRaw(SWITCH)!;
		expect(catchingBranch(draft, 1, 'look', 'suit', vars)).toBe(0);
		expect(catchingBranch(draft, 1, 'look', 'gown', vars)).toBeNull();
		expect(catchingBranch(draft, 3, 'look', 'jeans', vars)).toBe(1);
		expect(catchingBranch(draft, 3, 'look', 'boots', vars)).toBe(2);
		expect(catchingBranch(draft, 0, 'look', 'suit', vars)).toBeNull();
	});

	it('ignores branches that read another variable or are unfinished', () => {
		const draft = draftFromRaw('<<if $mood = "calm">>a<<else if $look = "suit">>b<<end>>')!;
		expect(catchingBranch(draft, 2, 'look', 'suit', vars)).toBe(1);
		expect(catchingBranch(draft, 2, 'mood', 'calm', vars)).toBe(0);
		draft.branches[1].tests[0].strings = [''];
		expect(catchingBranch(draft, 2, 'look', 'suit', vars)).toBeNull();
	});

	it('starts a new branch on the first option nothing catches', () => {
		const draft = draftFromRaw('<<if $look = "suit">>a<<else>>b<<end>>')!;
		expect(firstUncoveredOption(draft, 'look', vars)).toBe('gown');
		expect(nextBranchTest(draft, 'look', vars).strings).toEqual(['gown']);
		const covered = draftFromRaw('<<if $look in "suit", "gown", "jeans">>a<<end>>')!;
		expect(firstUncoveredOption(covered, 'look', vars)).toBeNull();
		expect(nextBranchTest(covered, 'look', vars).strings).toEqual(['suit']);
		expect(nextBranchTest(covered, 'age', vars)).toEqual(defaultTest('age', vars));
	});
});

describe('branch problem messages', () => {
	it('drops the branch name where the message sits inside its branch', () => {
		const draft: ConditionDraft = {
			branches: [
				{ kind: 'if', joiner: 'and', tests: [{ var: 'mood', op: '=', number: '0', low: '0', high: '0', strings: [''] }], body: 'a' },
				{ kind: 'else', joiner: 'and', tests: [], body: '<<if $age > 1>>b<<end>>' }
			]
		};
		const problems = checkDraft(draft, variables).problems;
		expect(problems.map((p) => branchProblemText(p, draft, variables))).toEqual([
			'Pick a value for $mood.',
			"This text contains another condition. Conditions can't be nested, so add an Else if instead."
		]);
		expect(branchProblemText({ kind: 'tests', branch: 0 }, draft, variables)).toBe(`Needs 1 to ${MAX_TESTS} tests.`);
		expect(branchProblemText({ kind: 'grammar' }, draft, variables)).toBe(draftProblemText({ kind: 'grammar' }, draft, variables));
	});
});
