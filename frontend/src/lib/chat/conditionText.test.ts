import { describe, it, expect } from 'vitest';
import {
	canApplySegmentAction,
	conditionTagProblems,
	formatNumberVariableSummary,
	hasConditionBlocks,
	parseNumberVariableSummary,
	splitConditionText
} from './conditionText';
import { buildApprovalDiff } from './approvalPreview';
import { collapsedSummary } from './actionCard';
import { MAX_BLOCKS, MAX_BRANCHES, MAX_TESTS } from '$lib/utils/promptConditions';
import type { ToolExecution } from '$lib/types/chat';

const BLOCK = '<<if $age > 65>>grey hair<<else>>auburn hair<<end>>';

describe('splitConditionText', () => {
	it('returns plain text untouched', () => {
		expect(splitConditionText('woman, soft light')).toEqual([{ kind: 'text', text: 'woman, soft light' }]);
	});

	it('splits text around a condition block', () => {
		expect(splitConditionText(`woman, ${BLOCK}, soft light`)).toEqual([
			{ kind: 'text', text: 'woman, ' },
			{ kind: 'condition', raw: BLOCK },
			{ kind: 'text', text: ', soft light' }
		]);
	});

	it('splits two blocks and keeps the text between them', () => {
		const parts = splitConditionText(`${BLOCK} and ${BLOCK}`);
		expect(parts.map((p) => p.kind)).toEqual(['condition', 'text', 'condition']);
	});

	it('leaves look-alike tags as text', () => {
		expect(hasConditionBlocks('<lora:x:1> <<hello>> <Picture 1>')).toBe(false);
		expect(hasConditionBlocks(BLOCK)).toBe(true);
	});
});

describe('conditionTagProblems', () => {
	it('accepts well formed blocks and tagless text', () => {
		expect(conditionTagProblems(BLOCK)).toEqual([]);
		expect(conditionTagProblems('plain')).toEqual([]);
	});

	it.each([
		['unclosed', 'woman <<if $age > 65>>grey hair'],
		['stray', 'woman <<else>> grey<<end>>'],
		['nested', '<<if $a > 1>>x<<if $b > 1>>y<<end>><<end>>'],
		['bad_head', '<<if $age>65>>x<<end>>'],
		['bad_head', '<<if $age > 1 and $b = "x" or $c < 2>>x<<end>>']
	])('reports %s with the backend error kind', (kind, text) => {
		const problems = conditionTagProblems(text);
		expect(problems.map((p) => p.kind)).toContain(kind);
		expect(problems.every((p) => p.message.length > 10)).toBe(true);
	});

	it('names the current caps when a block is over them', () => {
		const over = `<<if $x = 1>>a${Array.from({ length: MAX_BRANCHES }, (_, n) => `<<else if $x = ${n}>>b`).join('')}<<end>>`;
		const limit = conditionTagProblems(over).find((p) => p.kind === 'limit')!;
		expect(limit.message).toBe(
			`A block is over the limits: ${MAX_TESTS} tests per head, ${MAX_BRANCHES} branches, ${MAX_BLOCKS} blocks.`
		);
	});

	it('does not repeat the same problem', () => {
		const problems = conditionTagProblems('<<end>>');
		expect(new Set(problems.map((p) => `${p.kind}:${p.offset}`)).size).toBe(problems.length);
	});
});

describe('number variable summaries', () => {
	it('parses the backend description for each mode', () => {
		expect(parseNumberVariableSummary('number from 18 to 90 in steps of 1 — pinned to 70')).toEqual({
			min: '18',
			max: '90',
			step: '1',
			mode: 'pin',
			pinned: '70'
		});
		expect(parseNumberVariableSummary('number from 0.5 to 1.5 in steps of 0.25 — re-rolls independently per image')?.mode).toBe('per-image');
		expect(parseNumberVariableSummary('number from 5 to 5 — shuffles each generation')).toMatchObject({ step: null, mode: 'shuffle' });
	});

	it('ignores text and choice descriptions', () => {
		expect(parseNumberVariableSummary('one of a, b — shuffles each generation')).toBeNull();
		expect(parseNumberVariableSummary('not set')).toBeNull();
	});

	it('formats a short line for the collapsed card', () => {
		const summary = parseNumberVariableSummary('number from 18 to 90 in steps of 1 — pinned to 70')!;
		expect(formatNumberVariableSummary(summary)).toBe('18–90 step 1, pinned 70');
	});
});

function variableExecution(newValue: string, oldValue = 'not set'): ToolExecution {
	return {
		tool_name: 'manage_prompt_variables',
		arguments: {},
		result: {
			success: true,
			data: JSON.stringify({
				status: 'pending_approval',
				proposed_changes: [{ field_name: '${age}', old_value: oldValue, new_value: newValue }]
			})
		},
		duration_ms: 0,
		approval_status: 'pending'
	} as ToolExecution;
}

describe('variable diff rows', () => {
	it('carries the parsed number summary on both sides', () => {
		const rows = buildApprovalDiff(
			variableExecution(
				'number from 18 to 90 in steps of 1 — pinned to 70',
				'number from 18 to 90 in steps of 1 — shuffles each generation'
			)
		)!;
		expect(rows[0].newNumber).toMatchObject({ mode: 'pin', pinned: '70' });
		expect(rows[0].oldNumber).toMatchObject({ mode: 'shuffle' });
	});

	it('leaves other variable rows as plain text', () => {
		const rows = buildApprovalDiff(variableExecution('grey hair'))!;
		expect(rows[0].newNumber).toBeUndefined();
		expect(rows[0].newValue).toBe('grey hair');
	});

	it('shortens the collapsed summary for a number variable', () => {
		const text = collapsedSummary(variableExecution('number from 18 to 90 in steps of 1 — pinned to 70'), 'Variables');
		expect(text).toBe('${age} 18–90 step 1, pinned 70');
	});
});

describe('canApplySegmentAction', () => {
	it('blocks a malformed update_segment', () => {
		expect(canApplySegmentAction({ type: 'update_segment', content: 'a <<if $age > 65>>grey' })).toBe(false);
	});

	it('allows a valid or tagless update_segment', () => {
		expect(canApplySegmentAction({ type: 'update_segment', content: `a ${BLOCK}` })).toBe(true);
		expect(canApplySegmentAction({ type: 'add_segment', content: 'plain' })).toBe(true);
	});

	it('allows delete_segment whatever the content', () => {
		expect(canApplySegmentAction({ type: 'delete_segment', content: '<<end>>' })).toBe(true);
	});
});
