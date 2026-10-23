import { MAX_BLOCKS, MAX_BRANCHES, MAX_TESTS, parseTemplate, type ConditionErrorKind } from '$lib/utils/promptConditions';

export type ConditionPart = { kind: 'text'; text: string } | { kind: 'condition'; raw: string };

export interface ConditionTagProblem {
	kind: ConditionErrorKind;
	offset: number;
	message: string;
}

const PROBLEM_MESSAGE: Record<ConditionErrorKind, string> = {
	malformed_tag: 'A tag is not a valid <<if …>>, <<else if …>>, <<else>> or <<end>>.',
	nested: 'A condition sits inside another condition. Nesting is not supported; use <<else if …>>.',
	stray: 'An <<else>>, <<else if …>> or <<end>> has no open <<if …>>.',
	unclosed: 'An <<if …>> never reaches <<end>>.',
	limit: `A block is over the limits: ${MAX_TESTS} tests per head, ${MAX_BRANCHES} branches, ${MAX_BLOCKS} blocks.`,
	bad_head:
		'A condition head cannot be read: put spaces around every operator, quote text values, and use only and or or, never both.'
};

export function splitConditionText(text: string): ConditionPart[] {
	const parsed = text.includes('<<') ? parseTemplate(text) : null;
	if (!parsed || parsed.blocks.length === 0) return [{ kind: 'text', text }];
	const parts: ConditionPart[] = [];
	let cursor = 0;
	for (const block of [...parsed.blocks].sort((a, b) => a.start - b.start)) {
		if (block.start < cursor) continue;
		if (block.start > cursor) parts.push({ kind: 'text', text: text.slice(cursor, block.start) });
		parts.push({ kind: 'condition', raw: text.slice(block.start, block.end) });
		cursor = block.end;
	}
	if (cursor < text.length) parts.push({ kind: 'text', text: text.slice(cursor) });
	return parts;
}

export function hasConditionBlocks(text: string): boolean {
	return text.includes('<<if') && splitConditionText(text).some((part) => part.kind === 'condition');
}

export function conditionTagProblems(text: string): ConditionTagProblem[] {
	if (!text.includes('<<')) return [];
	const parsed = parseTemplate(text);
	if (!parsed) return [];
	const seen = new Set<string>();
	const problems: ConditionTagProblem[] = [];
	for (const error of parsed.errors) {
		const key = `${error.kind}:${error.offset}`;
		if (seen.has(key)) continue;
		seen.add(key);
		problems.push({ kind: error.kind, offset: error.offset, message: PROBLEM_MESSAGE[error.kind] ?? error.kind });
	}
	return problems;
}

export type NumberVariableMode = 'shuffle' | 'pin' | 'per-image';

export interface NumberVariableSummary {
	min: string;
	max: string;
	step: string | null;
	mode: NumberVariableMode;
	pinned: string | null;
}

const NUMBER_SUMMARY =
	/^number from (-?[\d.]+) to (-?[\d.]+)(?: in steps of (-?[\d.]+))? — (?:(pinned to (-?[\d.]+))|(re-rolls independently per image)|(shuffles each generation))/;

export function parseNumberVariableSummary(text: string): NumberVariableSummary | null {
	const match = NUMBER_SUMMARY.exec(text.trim());
	if (!match) return null;
	return {
		min: match[1],
		max: match[2],
		step: match[3] ?? null,
		mode: match[4] ? 'pin' : match[6] ? 'per-image' : 'shuffle',
		pinned: match[5] ?? null
	};
}

export const NUMBER_MODE_LABEL: Record<NumberVariableMode, string> = {
	shuffle: 'Shuffles each generation',
	pin: 'Pinned',
	'per-image': 'Re-rolls per image'
};

export function formatNumberVariableSummary(summary: NumberVariableSummary): string {
	const range = summary.min === summary.max ? summary.min : `${summary.min}–${summary.max}`;
	const step = summary.step ? ` step ${summary.step}` : '';
	const mode = summary.mode === 'pin' ? `pinned ${summary.pinned ?? summary.min}` : NUMBER_MODE_LABEL[summary.mode].toLowerCase();
	return `${range}${step}, ${mode}`;
}

export function canApplySegmentAction(action: { type: string; content: string }): boolean {
	return action.type === 'delete_segment' || conditionTagProblems(action.content).length === 0;
}
