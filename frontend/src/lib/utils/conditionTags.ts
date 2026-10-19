import {
	MAX_BRANCHES,
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

export function isValidString(text: string): boolean {
	return text.length >= 1 && text.length <= MAX_STRING_CHARS;
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
		const def = normalizeVariableDef(variables[name]);
		const first = def.type === 'choice' ? def.options.map((o) => optionText(o).trim()).find((o) => o) : undefined;
		test.strings = [first ?? ''];
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
	if (kind === 'choice' && (op === 'in' || op === 'not in') && next.strings.every((s) => !s)) {
		const def = normalizeVariableDef(variables[test.var]);
		const first = def.type === 'choice' ? def.options.map((o) => optionText(o).trim()).find((o) => o) : undefined;
		next.strings = [first ?? ''];
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

export function testFromDraft(draft: TestDraft, variables: VariablesMap): ConditionTest | null {
	const base = { var: draft.var, start: 0, end: 0 };
	const kind = conditionVarKind(draft.var, variables);
	const op = draft.op;
	if (op === 'is empty' || op === 'is not empty') return { ...base, kind: 'none', op };
	if (op === 'in' || op === 'not in') {
		if (kind === 'number') {
			if (!isValidNumberText(draft.low) || !isValidNumberText(draft.high)) return null;
			if (Number(draft.low) > Number(draft.high)) return null;
			return { ...base, kind: 'range', op, low: draft.low, high: draft.high };
		}
		const strings = draft.strings.filter((s) => s !== '');
		if (strings.length === 0 || strings.length > MAX_VALUES || !strings.every(isValidString)) return null;
		return { ...base, kind: 'strings', op, strings };
	}
	if (op === 'contains' || op === 'not contains') {
		const text = draft.strings[0] ?? '';
		return isValidString(text) ? { ...base, kind: 'string', op, strings: [text] } : null;
	}
	if (kind === 'number' || op === '<' || op === '<=' || op === '>' || op === '>=') {
		return isValidNumberText(draft.number) ? { ...base, kind: 'number', op, number: draft.number } : null;
	}
	const text = draft.strings[0] ?? '';
	return isValidString(text) ? { ...base, kind: 'string', op, strings: [text] } : null;
}

export function rawFromDraft(draft: ConditionDraft, variables: VariablesMap): string | null {
	const parts: string[] = [];
	for (const branch of draft.branches) {
		if (NESTED_TAG.test(branch.body)) return null;
		if (branch.kind === 'else') {
			parts.push(`<<else>>${branch.body}`);
			continue;
		}
		const tests = branch.tests.map((t) => testFromDraft(t, variables));
		if (tests.length === 0 || tests.length > MAX_TESTS || tests.some((t) => t === null)) return null;
		const head = formatCondition(tests as ConditionTest[], branch.joiner);
		parts.push(`<<${branch.kind === 'if' ? 'if' : 'else if'} ${head}>>${branch.body}`);
	}
	if (draft.branches.length > MAX_BRANCHES) return null;
	return `${parts.join('')}<<end>>`;
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
