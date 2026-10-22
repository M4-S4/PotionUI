import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { checkDraft, rawFromDraft, type BranchDraft, type TestDraft } from './conditionTags';
import { MAX_STRING_CHARS, renderConditions } from './promptConditions';
import type { VariablesMap } from './variableDefs';

interface CorpusCase {
	name: string;
	template: string;
	values: Record<string, string>;
	expected: string;
	errors: string[];
}

const corpusPath = resolve(__dirname, '../../../../tests/fixtures/prompt_conditions/evaluate.json');
const cases = new Map(
	(JSON.parse(readFileSync(corpusPath, 'utf8')) as { cases: CorpusCase[] }).cases.map((c) => [c.name, c])
);
const corpusCase = (name: string): CorpusCase => cases.get(name)!;

const equals = (option: string): TestDraft => ({
	var: 'look',
	op: '=',
	number: '0',
	low: '0',
	high: '0',
	strings: [option]
});

const branch = (kind: BranchDraft['kind'], body: string, tests: TestDraft[] = []): BranchDraft => ({
	kind,
	joiner: 'and',
	tests,
	body
});

const lookOf = (options: string[]): VariablesMap => ({
	look: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options }
});

describe('editor output for long choice options', () => {
	const switchCases = ['editor_switch_option_81_chars', 'editor_switch_option_200_chars', 'editor_switch_option_at_value_cap'].map(
		corpusCase
	);
	const options = switchCases.map((c) => c.values.look);

	it('covers options of 81, 200 and the value cap characters', () => {
		expect(options.map((o) => Array.from(o).length)).toEqual([81, 200, MAX_STRING_CHARS]);
	});

	it('writes a switch on them exactly as the shared corpus holds it', () => {
		const draft = {
			branches: [
				branch('if', 'A', [equals(options[0])]),
				branch('elseif', 'B', [equals(options[1])]),
				branch('elseif', 'C', [equals(options[2])]),
				branch('else', 'D')
			]
		};
		const raw = rawFromDraft(draft, lookOf(options));
		expect(raw).toBe(switchCases[0].template);
		switchCases.forEach((c) => {
			expect(renderConditions(raw!, c.values).text).toBe(c.expected);
		});
	});

	it('escapes quotes and backslashes in an option at the cap the way the corpus does', () => {
		const escaped = corpusCase('editor_escaped_option_at_value_cap');
		const option = escaped.values.look;
		const draft = { branches: [branch('if', 'A', [equals(option)]), branch('else', 'B')] };
		expect(rawFromDraft(draft, lookOf([option]))).toBe(escaped.template);
	});

	it('writes two options at the cap in one list', () => {
		const listed = corpusCase('editor_two_options_at_value_cap_in_one_list');
		const lists = Array.from(listed.template.matchAll(/"((?:[^"\\]|\\.)*)"/g), (m) => m[1].replace(/\\(.)/g, '$1'));
		const test = { ...equals(lists[0]), op: 'in' as const, strings: lists };
		const draft = { branches: [branch('if', 'A', [test]), branch('else', 'B')] };
		expect(rawFromDraft(draft, lookOf(lists))).toBe(listed.template);
	});

	it('refuses an option one over the cap, which the parser would read as a bad head', () => {
		const over = corpusCase('editor_option_one_over_value_cap');
		const option = over.values.look;
		const draft = { branches: [branch('if', 'A', [equals(option)]), branch('else', 'B')] };
		expect(checkDraft(draft, lookOf([option])).problems).toEqual([
			{ kind: 'value', branch: 0, test: 0, reason: 'too-long' }
		]);
		expect(over.errors).toEqual(['bad_head']);
	});
});
