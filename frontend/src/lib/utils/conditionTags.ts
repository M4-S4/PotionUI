import {
	MAX_BRANCHES,
	MAX_HEAD_CHARS,
	MAX_STRING_CHARS,
	MAX_TESTS,
	MAX_VALUES,
	chooseBranch,
	formatCondition,
	parseTemplate,
	type BranchKind,
	type ConditionBlock,
	type ConditionJoiner,
	type ConditionOp,
	type ConditionTest,
	type ParsedTemplate
} from './promptConditions';
import {
	clampNumberValue,
	normalizeVariableDef,
	numberGridOf,
	optionText,
	type ConditionReader,
	type ConditionReaders,
	type NumberTrackRange,
	type VariableRoll,
	type VariablesMap
} from './variableDefs';

export type ConditionVarKind = 'number' | 'choice' | 'text' | 'unknown';

export interface TestDraft {
	var: string;
	op: ConditionOp;
	number: string;
	low: string;
	high: string;
	strings: string[];
}

export interface BranchDraft {
	kind: BranchKind;
	joiner: ConditionJoiner;
	tests: TestDraft[];
	body: string;
}

export interface ConditionDraft {
	branches: BranchDraft[];
}

export type ValueProblem = 'empty' | 'too-long' | 'number' | 'reversed' | 'too-many';

export type DraftProblem =
	| { kind: 'value'; branch: number; test: number; reason: ValueProblem }
	| { kind: 'tests'; branch: number }
	| { kind: 'head'; branch: number }
	| { kind: 'nested'; branch: number }
	| { kind: 'branches' }
	| { kind: 'grammar' };

export interface DraftCheck {
	raw: string | null;
	problems: DraftProblem[];
}

export interface OperatorChoice {
	op: ConditionOp;
	label: string;
	glyph: string;
}

const NUMBER_OPERATORS: OperatorChoice[] = [
	{ op: '=', label: 'equals', glyph: '=' },
	{ op: '!=', label: 'not equal to', glyph: '≠' },
	{ op: '<', label: 'less than', glyph: '<' },
	{ op: '<=', label: 'at most', glyph: '≤' },
	{ op: '>', label: 'more than', glyph: '>' },
	{ op: '>=', label: 'at least', glyph: '≥' },
	{ op: 'in', label: 'between', glyph: '..' },
	{ op: 'not in', label: 'not between', glyph: '!..' }
];

const CHOICE_OPERATORS: OperatorChoice[] = [
	{ op: '=', label: 'is', glyph: '=' },
	{ op: '!=', label: 'is not', glyph: '≠' },
	{ op: 'in', label: 'is one of', glyph: '∈' },
	{ op: 'not in', label: 'is none of', glyph: '∉' },
	{ op: 'is empty', label: 'is empty', glyph: '∅' },
	{ op: 'is not empty', label: 'is not empty', glyph: '!∅' }
];

const TEXT_OPERATORS: OperatorChoice[] = [
	{ op: '=', label: 'is', glyph: '=' },
	{ op: '!=', label: 'is not', glyph: '≠' },
	{ op: 'contains', label: 'contains', glyph: '⊃' },
	{ op: 'not contains', label: 'does not contain', glyph: '⊅' },
	{ op: 'is empty', label: 'is empty', glyph: '∅' },
	{ op: 'is not empty', label: 'is not empty', glyph: '!∅' }
];

const NUMBER_PATTERN = /^-?\d+(\.\d+)?$/;
const NESTED_TAG = /<<\s*(if|else|end)/;

export function isConditionRaw(raw: string): boolean {
	return raw.startsWith('<<if');
}

export function conditionVarKind(name: string, variables: VariablesMap): ConditionVarKind {
	if (!(name in variables)) return 'unknown';
	return normalizeVariableDef(variables[name]).type;
}

export function operatorsFor(kind: ConditionVarKind): OperatorChoice[] {
	if (kind === 'number') return NUMBER_OPERATORS;
	if (kind === 'choice') return CHOICE_OPERATORS;
	return TEXT_OPERATORS;
}

export function operatorChoice(kind: ConditionVarKind, op: ConditionOp): OperatorChoice {
	return operatorsFor(kind).find((choice) => choice.op === op) ?? operatorsFor(kind)[0];
}

export function isValidNumberText(text: string): boolean {
	return NUMBER_PATTERN.test(text) && text.length <= 20;
}

function stringProblem(text: string): ValueProblem | null {
	if (text.length === 0) return 'empty';
	return Array.from(text).length > MAX_STRING_CHARS ? 'too-long' : null;
}

export function fitsCondition(text: string): boolean {
	return stringProblem(text) === null;
}

function firstOption(name: string, variables: VariablesMap): string {
	const def = normalizeVariableDef(variables[name]);
	if (def.type !== 'choice') return '';
	const options = def.options.map((o) => optionText(o).trim()).filter((o) => o);
	return options.find(fitsCondition) ?? options[0] ?? '';
}

function emptyTest(name: string): TestDraft {
	return { var: name, op: '=', number: '0', low: '0', high: '0', strings: [''] };
}

export function defaultTest(name: string, variables: VariablesMap): TestDraft {
	const kind = conditionVarKind(name, variables);
	const test = emptyTest(name);
	if (kind === 'number') {
		const def = normalizeVariableDef(variables[name]);
		if (def.type === 'number') {
			const grid = numberGridOf(def);
			const first = grid ? clampNumberValue({ ...def, value: def.min }) : '0';
			const last = grid ? clampNumberValue({ ...def, value: def.max }) : '0';
			test.op = '>=';
			test.number = first;
			test.low = first;
			test.high = last;
		}
	} else if (kind === 'choice') {
		test.strings = [firstOption(name, variables)];
	} else {
		test.op = 'is empty';
	}
	return test;
}

export function retargetTest(test: TestDraft, name: string, variables: VariablesMap): TestDraft {
	const kind = conditionVarKind(name, variables);
	const fresh = defaultTest(name, variables);
	const allowed = operatorsFor(kind).some((choice) => choice.op === test.op);
	return allowed && conditionVarKind(test.var, variables) === kind ? { ...test, var: name } : fresh;
}

export function setTestOperator(test: TestDraft, op: ConditionOp, variables: VariablesMap): TestDraft {
	const kind = conditionVarKind(test.var, variables);
	const next = { ...test, op };
	const listed = op === 'in' || op === 'not in';
	if (!listed && next.strings.length > 1) next.strings = [next.strings.find((s) => s) ?? ''];
	if (kind === 'choice' && op !== 'is empty' && op !== 'is not empty' && next.strings.every((s) => !s)) {
		next.strings = [firstOption(test.var, variables)];
	}
	return next;
}

function draftFromTest(test: ConditionTest): TestDraft {
	const draft = emptyTest(test.var);
	draft.op = test.op;
	switch (test.kind) {
		case 'number':
			draft.number = test.number;
			draft.low = test.number;
			draft.high = test.number;
			break;
		case 'range':
			draft.low = test.low;
			draft.high = test.high;
			draft.number = test.low;
			break;
		case 'string':
		case 'strings':
			draft.strings = [...test.strings];
			break;
		case 'none':
			break;
	}
	return draft;
}

export function draftFromBlock(block: ConditionBlock): ConditionDraft | null {
	const branches: BranchDraft[] = [];
	for (const branch of block.branches) {
		if (branch.kind === 'else') {
			branches.push({ kind: 'else', joiner: 'and', tests: [], body: branch.body });
			continue;
		}
		if (!branch.cond) return null;
		branches.push({
			kind: branch.kind,
			joiner: branch.cond.joiner,
			tests: branch.cond.tests.map(draftFromTest),
			body: branch.body
		});
	}
	return { branches };
}

export function parseConditionRaw(raw: string): ParsedTemplate | null {
	const parsed = parseTemplate(raw);
	if (!parsed || parsed.blocks.length !== 1 || parsed.errors.length > 0) return null;
	const block = parsed.blocks[0];
	return block && block.start === 0 && block.end === raw.length ? parsed : null;
}

export function draftFromRaw(raw: string): ConditionDraft | null {
	const parsed = parseConditionRaw(raw);
	return parsed ? draftFromBlock(parsed.blocks[0]) : null;
}

type TestRead = { test: ConditionTest; problem: null } | { test: null; problem: ValueProblem };

function readTest(draft: TestDraft, variables: VariablesMap): TestRead {
	const base = { var: draft.var, start: 0, end: 0 };
	const kind = conditionVarKind(draft.var, variables);
	const op = draft.op;
	const ok = (test: ConditionTest): TestRead => ({ test, problem: null });
	const fail = (problem: ValueProblem): TestRead => ({ test: null, problem });
	if (op === 'is empty' || op === 'is not empty') return ok({ ...base, kind: 'none', op });
	if (op === 'in' || op === 'not in') {
		if (kind === 'number') {
			if (!isValidNumberText(draft.low) || !isValidNumberText(draft.high)) return fail('number');
			if (Number(draft.low) > Number(draft.high)) return fail('reversed');
			return ok({ ...base, kind: 'range', op, low: draft.low, high: draft.high });
		}
		const strings = draft.strings.filter((s) => s !== '');
		if (strings.length === 0) return fail('empty');
		if (strings.length > MAX_VALUES) return fail('too-many');
		const problem = strings.map(stringProblem).find((p) => p !== null);
		return problem ? fail(problem) : ok({ ...base, kind: 'strings', op, strings });
	}
	if (op === 'contains' || op === 'not contains') {
		const text = draft.strings[0] ?? '';
		const problem = stringProblem(text);
		return problem ? fail(problem) : ok({ ...base, kind: 'string', op, strings: [text] });
	}
	if (kind === 'number' || op === '<' || op === '<=' || op === '>' || op === '>=') {
		return isValidNumberText(draft.number) ? ok({ ...base, kind: 'number', op, number: draft.number }) : fail('number');
	}
	const text = draft.strings[0] ?? '';
	const problem = stringProblem(text);
	return problem ? fail(problem) : ok({ ...base, kind: 'string', op, strings: [text] });
}

export function testFromDraft(draft: TestDraft, variables: VariablesMap): ConditionTest | null {
	return readTest(draft, variables).test;
}

export function checkDraft(draft: ConditionDraft, variables: VariablesMap): DraftCheck {
	const problems: DraftProblem[] = [];
	const parts: string[] = [];
	if (draft.branches.length > MAX_BRANCHES) problems.push({ kind: 'branches' });
	draft.branches.forEach((branch, index) => {
		if (NESTED_TAG.test(branch.body)) problems.push({ kind: 'nested', branch: index });
		if (branch.kind === 'else') {
			parts.push(`<<else>>${branch.body}`);
			return;
		}
		if (branch.tests.length === 0 || branch.tests.length > MAX_TESTS) {
			problems.push({ kind: 'tests', branch: index });
			return;
		}
		const tests: ConditionTest[] = [];
		branch.tests.forEach((draftTest, test) => {
			const read = readTest(draftTest, variables);
			if (read.test !== null) tests.push(read.test);
			else problems.push({ kind: 'value', branch: index, test, reason: read.problem });
		});
		if (tests.length < branch.tests.length) return;
		const head = formatCondition(tests, branch.joiner);
		if (Array.from(head).length > MAX_HEAD_CHARS) problems.push({ kind: 'head', branch: index });
		parts.push(`<<${branch.kind === 'if' ? 'if' : 'else if'} ${head}>>${branch.body}`);
	});
	if (problems.length > 0) return { raw: null, problems };
	const raw = `${parts.join('')}<<end>>`;
	return parseConditionRaw(raw) ? { raw, problems } : { raw: null, problems: [{ kind: 'grammar' }] };
}

export function rawFromDraft(draft: ConditionDraft, variables: VariablesMap): string | null {
	return checkDraft(draft, variables).raw;
}

export function nestedToElseIf(draft: ConditionDraft): ConditionDraft | null {
	const last = draft.branches[draft.branches.length - 1];
	if (!last || last.kind !== 'else') return null;
	const inner = draftFromRaw(last.body.trim());
	if (!inner) return null;
	const branches: BranchDraft[] = [
		...draft.branches.slice(0, -1),
		...inner.branches.map((b, i): BranchDraft => (i === 0 ? { ...b, kind: 'elseif' } : b))
	];
	return branches.length <= MAX_BRANCHES ? { branches } : null;
}

export function branchName(draft: ConditionDraft, index: number): string {
	const branch = draft.branches[index];
	if (!branch || branch.kind === 'if') return 'If';
	if (branch.kind === 'else') return 'Otherwise';
	if (draft.branches.filter((b) => b.kind === 'elseif').length < 2) return 'Else if';
	return `Else if ${draft.branches.slice(0, index + 1).filter((b) => b.kind === 'elseif').length}`;
}

function valueProblemText(reason: ValueProblem, test: TestDraft, variables: VariablesMap): string {
	const target = `$${test.var}`;
	switch (reason) {
		case 'empty':
			if (test.op === 'in' || test.op === 'not in') return `pick at least one value for ${target}.`;
			return conditionVarKind(test.var, variables) === 'choice'
				? `pick a value for ${target}.`
				: `type the text to compare ${target} with.`;
		case 'too-long':
			return conditionVarKind(test.var, variables) === 'choice'
				? `the option picked for ${target} is longer than ${MAX_STRING_CHARS} characters and can't be used in a condition.`
				: `the value for ${target} is longer than ${MAX_STRING_CHARS} characters and can't be used in a condition.`;
		case 'number':
			return `enter a number for ${target}.`;
		case 'reversed':
			return `the From number for ${target} is above To.`;
		case 'too-many':
			return `pick at most ${MAX_VALUES} values for ${target}.`;
	}
}

export function draftProblemText(problem: DraftProblem, draft: ConditionDraft, variables: VariablesMap): string {
	switch (problem.kind) {
		case 'branches':
			return `A condition can have at most ${MAX_BRANCHES} branches.`;
		case 'grammar':
			return "This condition can't be saved as it is. Shorten its text or check its variable names.";
		case 'nested':
			return `${branchName(draft, problem.branch)} text contains another condition. Conditions can't be nested, so add an Else if instead.`;
		case 'tests':
			return `${branchName(draft, problem.branch)} needs 1 to ${MAX_TESTS} tests.`;
		case 'head':
			return `${branchName(draft, problem.branch)} is too long to save. Remove a test or pick shorter values.`;
		case 'value':
			return `${branchName(draft, problem.branch)}: ${valueProblemText(problem.reason, draft.branches[problem.branch].tests[problem.test], variables)}`;
	}
}

export function defaultConditionRaw(variables: VariablesMap, body = ''): string {
	const name = Object.keys(variables)[0] ?? 'value';
	const test = defaultTest(name, variables);
	const raw = rawFromDraft(
		{ branches: [{ kind: 'if', joiner: 'and', tests: [test], body: body || 'text' }] },
		variables
	);
	return raw ?? `<<if $${name} is empty>>${body || 'text'}<<end>>`;
}

export function countConditionBlocks(text: string): number {
	if (!text.includes('<<')) return 0;
	return parseTemplate(text)?.blocks.length ?? 0;
}

export function testIssue(test: ConditionTest, variables: VariablesMap): string | null {
	const name = test.var;
	if (!(name in variables)) return `$${name} isn't a variable on this tab.`;
	const def = normalizeVariableDef(variables[name]);
	const numeric = test.kind === 'number' || test.kind === 'range';
	if (def.type === 'number') {
		if (!numeric && test.kind !== 'none') {
			return `${name} is a number now. Compare it with a number, or change it back to Choice.`;
		}
		const grid = numberGridOf(def);
		if (!grid || !numeric) return null;
		const lo = def.min;
		const hi = def.max;
		const never = (() => {
			if (test.kind === 'range') {
				if (test.op !== 'in') return false;
				return Number(test.high) < lo || Number(test.low) > hi;
			}
			const n = Number(test.number);
			switch (test.op) {
				case '>':
					return n >= hi;
				case '>=':
					return n > hi;
				case '<':
					return n <= lo;
				case '<=':
					return n < lo;
				case '=':
					return n < lo || n > hi;
				default:
					return false;
			}
		})();
		return never ? `${name} only goes between ${lo} and ${hi}, so this branch can never match.` : null;
	}
	if (numeric) {
		return `${name} is a ${def.type} variable now. Compare it with text, or change it back to Number.`;
	}
	const listed = test.kind === 'strings' || (test.kind === 'string' && (test.op === '=' || test.op === '!='));
	if (def.type === 'choice' && listed) {
		const options = new Set(def.options.map((o) => optionText(o).trim()));
		const gone = test.strings.find((s) => !options.has(s));
		return gone === undefined ? null : `${gone} is no longer an option of $${name}.`;
	}
	return null;
}

export function conditionIssue(raw: string, variables: VariablesMap): string | null {
	const parsed = parseConditionRaw(raw);
	if (!parsed) return 'This condition is malformed. Edit the text or remove it.';
	for (const branch of parsed.blocks[0].branches) {
		for (const test of branch.cond?.tests ?? []) {
			const issue = testIssue(test, variables);
			if (issue) return issue;
		}
	}
	return null;
}

export interface ConditionSummary {
	head: string;
	then: string;
	otherwise: string | null;
	extra: number;
}

function headText(cond: NonNullable<ConditionBlock['branches'][number]['cond']>): string {
	return formatCondition(cond).replace(/\$/g, '');
}

export function summarizeCondition(raw: string): ConditionSummary | null {
	const parsed = parseConditionRaw(raw);
	if (!parsed) return null;
	const branches = parsed.blocks[0].branches;
	const first = branches[0];
	if (!first?.cond) return null;
	const last = branches[branches.length - 1];
	return {
		head: headText(first.cond),
		then: first.body.trim(),
		otherwise: last && last.kind === 'else' ? last.body.trim() : null,
		extra: branches.filter((b) => b.kind === 'elseif').length
	};
}

export function conditionTooltip(raw: string): string {
	const parsed = parseConditionRaw(raw);
	if (!parsed) return raw;
	return parsed.blocks[0].branches
		.map((b) => {
			const lead = b.kind === 'else' ? 'otherwise' : `${b.kind === 'if' ? 'if' : 'else if'} ${b.cond ? headText(b.cond) : ''}`;
			return `${lead} → ${b.body.trim() || 'nothing'}`;
		})
		.join('\n');
}

function splitFor(test: ConditionTest, kind: BranchKind): NumberTrackRange | null {
	const lead = kind === 'if' ? 'IF' : 'ELSE IF';
	if (test.kind === 'range' && test.op === 'in') {
		return { from: Number(test.low), to: Number(test.high), label: `${lead} ${test.low}..${test.high}`, align: 'start' };
	}
	if (test.kind !== 'number') return null;
	const n = Number(test.number);
	switch (test.op) {
		case '>':
		case '>=':
			return { from: n, to: Infinity, label: `${lead} ${test.op} ${test.number}`, align: 'end' };
		case '<':
		case '<=':
			return { from: -Infinity, to: n, label: `${lead} ${test.op} ${test.number}`, align: 'start' };
		default:
			return null;
	}
}

export function conditionReadersFor(texts: readonly string[]): ConditionReaders {
	const readers: ConditionReaders = {};
	texts.forEach((text, index) => {
		if (!text.includes('<<')) return;
		const parsed = parseTemplate(text);
		if (!parsed) return;
		const segment = `segment ${String(index + 1).padStart(2, '0')}`;
		for (const block of parsed.blocks) {
			const seen = new Set<string>();
			for (const branch of block.branches) {
				if (!branch.cond) continue;
				for (const test of branch.cond.tests) {
					if (seen.has(test.var)) continue;
					seen.add(test.var);
					const reader: ConditionReader = { label: headText(branch.cond), segment };
					const splits = branch.cond.tests
						.filter((t) => t.var === test.var)
						.map((t) => splitFor(t, branch.kind))
						.filter((split): split is NumberTrackRange => split !== null);
					if (splits.length > 0) reader.splits = splits;
					(readers[test.var] ??= []).push(reader);
				}
			}
		}
	});
	return readers;
}

export function conditionValues(
	variables: VariablesMap,
	rolls: Record<string, VariableRoll> = {}
): Record<string, string> {
	const values: Record<string, string> = {};
	for (const [name, stored] of Object.entries(variables)) {
		const def = normalizeVariableDef(stored);
		if (def.type === 'text') {
			values[name] = def.value.trim();
			continue;
		}
		if (def.type === 'number') {
			if (def.mode === 'pin') values[name] = clampNumberValue(def);
			else if (def.mode === 'shuffle' && rolls[name]?.value) values[name] = rolls[name].value;
			continue;
		}
		if (def.mode === 'pin' && def.pinnedIndex !== null) {
			const pinned = def.options[def.pinnedIndex];
			if (pinned !== undefined) values[name] = optionText(pinned).trim();
		} else if (def.mode === 'shuffle' && rolls[name]?.value) {
			values[name] = rolls[name].value.trim();
		}
	}
	return values;
}

export interface LastRun {
	reads: { name: string; value: string | null }[];
	position: number | null;
	body: string;
	settled: boolean;
}

export function lastRunFor(raw: string, values: Record<string, string>): LastRun | null {
	const parsed = parseConditionRaw(raw);
	if (!parsed) return null;
	const block = parsed.blocks[0];
	const names = new Set<string>();
	for (const branch of block.branches) for (const test of branch.cond?.tests ?? []) names.add(test.var);
	const reads = [...names].map((name) => ({ name, value: name in values ? values[name] : null }));
	const { position, body } = chooseBranch(block, values);
	return { reads, position, body, settled: reads.every((r) => r.value !== null) };
}

export function conditionTagWarnings(text: string): { start: number; end: number; tone: 'warning' }[] {
	if (!text.includes('<<')) return [];
	const parsed = parseTemplate(text);
	if (!parsed) return [];
	return parsed.errors.map((error) => {
		const close = text.indexOf('>>', error.offset);
		const end = close === -1 ? Math.min(text.length, error.offset + 4) : close + 2;
		return { start: error.offset, end: Math.max(end, error.offset + 1), tone: 'warning' as const };
	});
}

const PLAIN_NUMBER_OPS: Record<string, string> = {
	'=': 'equals',
	'!=': 'is not',
	'<': 'is less than',
	'<=': 'is at most',
	'>': 'is more than',
	'>=': 'is at least'
};

function plainTest(test: ConditionTest): string {
	const lead = test.var;
	switch (test.kind) {
		case 'none':
			return `${lead} ${test.op}`;
		case 'number':
			return `${lead} ${PLAIN_NUMBER_OPS[test.op] ?? test.op} ${test.number}`;
		case 'range':
			return `${lead} ${test.op === 'in' ? 'is between' : 'is not between'} ${test.low} and ${test.high}`;
		case 'string': {
			const word = test.op === '=' ? 'is' : test.op === '!=' ? 'is not' : test.op === 'contains' ? 'contains' : 'does not contain';
			return `${lead} ${word} ${test.strings[0]}`;
		}
		case 'strings':
			return `${lead} ${test.op === 'in' ? 'is one of' : 'is none of'} ${test.strings.join(', ')}`;
	}
}

export function plainConditionSummary(raw: string): string {
	const parsed = parseConditionRaw(raw);
	const branches = parsed?.blocks[0].branches ?? [];
	const cond = branches[0]?.cond;
	if (!cond) return 'Condition';
	const lead = `When ${cond.tests.map(plainTest).join(` ${cond.joiner} `)}`;
	const more = branches.length - 1;
	return more > 0 ? `${lead} +${more} more` : lead;
}
