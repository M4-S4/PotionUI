import type { Segment } from '$lib/types/segments';
import { applySegmentAffixes, isSegmentEnabled } from './richSegments';
import { richTextToPlainText } from './richTextUtils';
import { parseVariableUsageTokens } from './promptVariables';
import { numberVariableUsages, type VariableUsageNumber } from './variableDefs';

const CHIP_MARK = '\u0001';

interface FlattenedPiece {
	text: string;
	bodyStart: number;
	bodyEnd: number;
}

function flattenPiece(segment: Segment): FlattenedPiece {
	if (!isSegmentEnabled(segment)) return { text: '', bodyStart: 0, bodyEnd: 0 };
	const chips = segment.chips || {};
	const markedChips = Object.fromEntries(
		Object.entries(chips).map(([id, chip]) => [id, { ...chip, value: (chip.value ?? '').split('${').join(`${CHIP_MARK}\${`) }])
	);
	const content = segment.content || '';
	const resolved = Object.keys(chips).length ? richTextToPlainText(content, markedChips) : content;
	const body = resolved.trim();
	if (!body.split(CHIP_MARK).join('').trim()) return { text: '', bodyStart: 0, bodyEnd: 0 };
	const text = applySegmentAffixes(segment, body);
	const prefixLength = text.length - body.length - (segment.suffix ?? '').length;
	return { text, bodyStart: prefixLength, bodyEnd: prefixLength + body.length };
}

export function segmentUsageNumbers(segments: Segment[], negativeSegments: Segment[] = []): VariableUsageNumber[][] {
	const pieces = [...segments, ...negativeSegments].map(flattenPiece);
	const numbered = numberVariableUsages(pieces.map((piece) => piece.text));
	return pieces.map((piece, index) => {
		const variableTokens = parseVariableUsageTokens(piece.text).filter((token) => token.type === 'variable');
		return numbered[index].filter((_, position) => {
			const token = variableTokens[position];
			return (
				token.start >= piece.bodyStart &&
				token.end <= piece.bodyEnd &&
				piece.text[token.start - 1] !== CHIP_MARK
			);
		});
	});
}
