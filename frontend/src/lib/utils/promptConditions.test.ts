import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
	MAX_BLOCKS,
	MAX_BRANCHES,
	MAX_HEAD_CHARS,
	MAX_STRING_CHARS,
	MAX_TESTS,
	chooseBranch,
	conditionReads,
	formatCondition,
	parseCondition,
	parseTemplate,
	renderConditions,
	sameCondition,
	splice
} from './promptConditions';

interface CorpusCase {
	name: string;
	template: string;
	values: Record<string, string>;
	expected: string;
	errors: string[];
}

const corpusPath = resolve(__dirname, '../../../../tests/fixtures/prompt_conditions/evaluate.json');
const corpus = JSON.parse(readFileSync(corpusPath, 'utf8')) as { cases: CorpusCase[] };

describe('prompt conditions corpus parity', () => {
	it('loads the full corpus', () => {
		expect(corpus.cases.length).toBeGreaterThan(0);
	});

	it.each(corpus.cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
		const result = renderConditions(c.template, c.values);
		expect(result.text).toBe(c.expected);
		expect(result.errors.map((e) => e.kind)).toEqual(c.errors);
	});
});

describe('decimal-exact comparison', () => {
	const run = (op: string, target: string, value: string) =>
		renderConditions(`<<if $n ${op} ${target}>>Y<<else>>N<<end>>`, { n: value }).text;

	it('compares without float rounding', () => {
		expect(run('=', '0.3', '0.30')).toBe('Y');
		expect(run('=', '0.3', String(0.1 + 0.2))).toBe('N');
		expect(run('<', '0.3', '0.29999999999999999999')).toBe('Y');
		expect(run('=', '1', '1.000')).toBe('Y');
		expect(run('>', '-0.5', '-0.4')).toBe('Y');
	});

	it('treats non-numeric values as false', () => {
		expect(run('=', '1', 'abc')).toBe('N');
		expect(run('!=', '1', 'abc')).toBe('N');
	});
});

describe('structure', () => {
	it('exposes blocks, branches and test spans', () => {
		const template = 'x <<if $a = "1" and $b in 1..5>>A<<else>>B<<end>>';
		const parsed = parseTemplate(template);
		expect(parsed).not.toBeNull();
		const block = parsed!.blocks[0]!;
		expect(template.slice(block.start, block.end)).toBe(template.slice(2));
		const [ifBranch, elseBranch] = block.branches;
		expect(ifBranch!.kind).toBe('if');
		expect(elseBranch!.kind).toBe('else');
		const tests = ifBranch!.cond!.tests;
		expect(tests.map((t) => template.slice(t.start, t.end))).toEqual(['$a = "1"', '$b in 1..5']);
		expect(template.slice(ifBranch!.bodyStart, ifBranch!.bodyEnd)).toBe('A');
	});
});

describe('formatCondition round trip', () => {
	const heads = [
		'$a = "x"',
		'$a != "x"',
		'$n = 3',
		'$n != -2.50',
		'$n < 1 and $n >= 0.5',
		'$n <= 10 or $m > 2',
		'$n in 1..5',
		'$n not in -3.5..-1',
		'$s in "a", "b", "c"',
		'$s not in "a"',
		'$s contains "red"',
		'$s not contains "red" and $t is empty',
		'$s is not empty or $t is empty',
		'$s = "say \\"hi\\" \\\\ now"'
	];

	it.each(heads)('%s', (head) => {
		const parsed = parseCondition(head);
		expect(parsed).not.toBeNull();
		const text = formatCondition(parsed!);
		expect(sameCondition(parseCondition(text), parsed)).toBe(true);
		expect(formatCondition(parseCondition(text)!)).toBe(text);
	});

	it('formats a bare test list with a joiner', () => {
		const parsed = parseCondition('$a = 1 or $b = 2')!;
		expect(formatCondition(parsed.tests, 'or')).toBe('$a = 1 or $b = 2');
	});

	it('rejects bad heads', () => {
		for (const bad of ['$a', '$a = ', '$a in 5..1', '$a = 1 and $b = 2 or $c = 3', 'a = 1']) {
			expect(parseCondition(bad)).toBeNull();
		}
	});
});

describe('limits', () => {
	const errorKinds = (text: string) => renderConditions(text, { x: 'y' }).errors.map((e) => e.kind);

	it('holds as many tests in a head as the cap and no more', () => {
		const head = (count: number, joiner: string) => Array.from({ length: count }, () => '$a = 1').join(` ${joiner} `);
		for (const joiner of ['and', 'or']) {
			expect(parseCondition(head(MAX_TESTS, joiner))!.tests).toHaveLength(MAX_TESTS);
			expect(parseCondition(head(MAX_TESTS + 1, joiner))).toBeNull();
		}
	});

	it('holds as many branches in a block as the cap and no more', () => {
		const withElseIfs = (count: number, tail: string) =>
			`<<if $x = 1>>a${Array.from({ length: count }, (_, n) => `<<else if $x = ${n}>>b`).join('')}${tail}<<end>>`;
		expect(parseTemplate(withElseIfs(MAX_BRANCHES - 1, ''))!.blocks).toHaveLength(1);
		expect(parseTemplate(withElseIfs(MAX_BRANCHES - 2, '<<else>>c'))!.blocks).toHaveLength(1);
		const over = parseTemplate(withElseIfs(MAX_BRANCHES, ''))!;
		expect(over.blocks).toHaveLength(0);
		expect(over.errors.map((e) => e.kind)).toContain('limit');
		expect(over.skeleton).not.toContain('<<');
		expect(parseTemplate(withElseIfs(MAX_BRANCHES - 1, '<<else>>c'))!.blocks).toHaveLength(0);
	});

	it('holds as many blocks as the cap and no more', () => {
		const one = '[<<if $x = "y">>A<<end>>]';
		const full = parseTemplate(one.repeat(MAX_BLOCKS))!;
		expect(full.blocks).toHaveLength(MAX_BLOCKS);
		expect(full.errors).toEqual([]);
		const over = one.repeat(MAX_BLOCKS + 1);
		expect(parseTemplate(over)!.blocks).toHaveLength(MAX_BLOCKS);
		expect(errorKinds(over)).toContain('limit');
		expect(renderConditions(over, { x: 'y' }).text).toBe('[A]'.repeat(MAX_BLOCKS + 1));
	});

	it('holds a head of the cap in characters and no more', () => {
		const block = (length: number) => {
			const head = '$x in "v", "w"';
			return `<<if ${head}${' '.repeat(length - head.length)}>>A<<else>>B<<end>>`;
		};
		expect(renderConditions(block(MAX_HEAD_CHARS), { x: 'v' })).toEqual({ text: 'A', errors: [] });
		const over = renderConditions(block(MAX_HEAD_CHARS + 1), { x: 'v' });
		expect(over.errors.map((e) => e.kind)).toContain('malformed_tag');
		expect(over.text).not.toContain('<<if');
	});

	it('holds a quoted value of the cap and no more', () => {
		const quoted = (length: number) => `$a = "${'x'.repeat(length)}"`;
		expect(parseCondition(quoted(MAX_STRING_CHARS))).not.toBeNull();
		expect(parseCondition(quoted(MAX_STRING_CHARS + 1))).toBeNull();
		expect(parseCondition('$a = ""')).toBeNull();
	});
});

describe('adversarial input', () => {
	const cases: Record<string, string> = {
		repeatedIfs: '<<if $a = "x">>y '.repeat(60_000),
		repeatedBlocks: '<<if $a = "x">>y<<end>>'.repeat(40_000),
		openers: '<<'.repeat(500_000),
		longHeads: `<<if ${'"'.repeat(300)}`.repeat(3_000),
		elseChain: `<<if $a = "1">>${'<<else if $a = "2">>z'.repeat(40_000)}`,
		tailTags: `${'a '.repeat(30_000)}${'<<end>>'.repeat(120_000)}`
	};

	it.each(Object.entries(cases))('%s stays bounded', (_name, input) => {
		expect(input.length).toBeGreaterThan(200_000);
		const started = performance.now();
		const result = renderConditions(input, { a: 'x' });
		const parsed = parseTemplate(input);
		const elapsed = performance.now() - started;
		expect(typeof result.text).toBe('string');
		expect(parsed === null || parsed.blocks.length <= MAX_BLOCKS).toBe(true);
		expect(elapsed).toBeLessThan(2000);
	});

	it('handles a 1 MB template', () => {
		const input = `${'x'.repeat(1_000_000)}<<if $a = "x">>y<<end>>`;
		const started = performance.now();
		const result = renderConditions(input, { a: 'x' });
		expect(performance.now() - started).toBeLessThan(2000);
		expect(result.text.endsWith('y')).toBe(true);
	});
});

describe('helpers', () => {
	const template =
		'<<if $a = "1" and $b > 2>>A<<else if $c in "x", "y" or $d is empty>>B ${x}<<else>>C ${x}<<end>>';

	it('conditionReads returns exactly the variables the heads read', () => {
		const parsed = parseTemplate(template)!;
		expect(parsed.errors).toEqual([]);
		expect(conditionReads(parsed)).toEqual(new Set(['a', 'b', 'c', 'd']));
	});

	it('chooseBranch picks the first match, then else, then null', () => {
		const block = parseTemplate(template)!.blocks[0]!;
		expect(chooseBranch(block, { a: '1', b: '3', c: 'x' })).toEqual({ position: 0, body: 'A' });
		expect(chooseBranch(block, { a: '0', c: 'y' })).toEqual({ position: 1, body: 'B ${x}' });
		expect(chooseBranch(block, {})).toEqual({ position: 2, body: 'C ${x}' });
		const noElse = parseTemplate('<<if $a = "1">>A<<end>>')!.blocks[0]!;
		expect(chooseBranch(noElse, { a: '2' })).toEqual({ position: null, body: '' });
	});

	it('splice with an empty body cleans up the comma hole', () => {
		expect(splice('red, \uE0100\uE011, blue', [''])).toBe('red, blue');
		expect(splice('red, blue, \uE0100\uE011', [''])).toBe('red, blue');
		expect(splice('red, \uE0100\uE011, blue', ['green'])).toBe('red, green, blue');
	});
});
