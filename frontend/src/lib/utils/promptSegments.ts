import type { ChipData, Segment } from '$lib/types/segments';
import {
	createBlankEditorSegment,
	ensureSegmentList,
	flattenRichSegments,
	isPristineBlankSegment,
	type SegmentJoin
} from '$lib/utils/richSegments';
import { parseChipsFromText } from '$lib/utils/chipParser';
import { logger } from '$lib/utils/logger';
import { deriveResourcesFromText, textHasResourceMarkers } from '$lib/utils/promptResources';
import { convertResourceTokens, type ResourceTokenContext } from '$lib/utils/promptResourceTokens';

/** Resolve enabled segments; collapsed state is presentation-only. */
export function resolvePromptSegments(segments: Segment[] = [], join: SegmentJoin = 'space'): string {
	return flattenRichSegments(segments, join);
}

/**
 * Locates the segment an LLM-proposed `<tool_action type="update_segment">`
 * targets: match by id first (segments can be reordered between the tool
 * call and the user clicking Apply), falling back to the index the LLM saw.
 * Returns -1 if neither resolves — callers should treat that as a no-op
 * rather than writing to the wrong segment.
 */
export function locateSegmentIndex(
	segments: Segment[],
	target: { segmentId: string; segmentIndex: number }
): number {
	const byId = segments.findIndex((s) => s.id === target.segmentId);
	if (byId !== -1) return byId;
	if (target.segmentIndex >= 0 && target.segmentIndex < segments.length) return target.segmentIndex;
	return -1;
}

/**
 * Re-applies the user's prior chip choices (value/shuffle/autoRegen) onto
 * freshly re-parsed chips for `#category.path` markers the LLM kept
 * unchanged, matched by category path. Each old chip is consumed at most
 * once so duplicate markers of the same category don't all inherit the same
 * prior value. Markers the LLM introduced (no category match) keep the
 * freshly parsed chip as-is.
 */
export function mergeChipSelections(
	oldChips: Record<string, ChipData>,
	newChips: Record<string, ChipData>
): Record<string, ChipData> {
	const merged = { ...newChips };
	const consumed = new Set<string>();
	for (const [chipId, chip] of Object.entries(merged)) {
		const match = Object.entries(oldChips).find(
			([oldId, old]) => !consumed.has(oldId) && old.categoryPath === chip.categoryPath
		);
		if (match) {
			consumed.add(match[0]);
			const old = match[1];
			merged[chipId] = {
				...chip,
				valueId: old.valueId,
				label: old.label,
				value: old.value,
				shuffle: old.shuffle,
				autoRegen: old.autoRegen
			};
		}
	}
	return merged;
}

/**
 * Applies one LLM-proposed segment update — resolve, hydrate `#category.path`
 * markers into chips (carrying forward the user's prior chip choices for
 * markers the LLM kept), write the new content — the shared core behind both
 * the legacy `<tool_action type="update_segment">` card apply and the
 * `update_segment` tool's `apply_segment_updates` result. Returns null when
 * the target segment can't be resolved.
 */
export async function applySegmentUpdate(
	sourceSegments: Segment[],
	target: { segmentId: string; segmentIndex: number; content: string },
	resourceContext?: ResourceTokenContext
): Promise<{ segments: Segment[]; index: number } | null> {
	const idx = locateSegmentIndex(sourceSegments, target);
	if (idx === -1) return null;

	const content = resourceContext?.specs.length
		? convertResourceTokens(target.content, resourceContext.specs, resourceContext.formValues).text
		: target.content;

	let newChips: Record<string, ChipData> = {};
	try {
		newChips = await parseChipsFromText(content, { shuffleCategoryChips: true });
	} catch (err) {
		logger.error('Failed to parse chips from applied segment content:', err);
	}

	const mergedChips = mergeChipSelections(sourceSegments[idx].chips || {}, newChips);
	const previousResources = sourceSegments[idx].resources;
	const segments = [...sourceSegments];
	segments[idx] = { ...segments[idx], content, chips: mergedChips };
	if (previousResources || textHasResourceMarkers(content)) {
		segments[idx].resources = deriveResourcesFromText(content, previousResources || {});
	}
	return { segments, index: idx };
}

export type SegmentListKey = 'positive' | 'negative';

export interface SegmentLists {
	positive: Segment[];
	negative: Segment[];
}

export interface SegmentRemoval {
	list: SegmentListKey;
	index: number;
	segment: Segment;
	previousId: string | null;
	nextId: string | null;
}

export function locateSegmentInLists(
	lists: SegmentLists,
	target: { segmentId: string; segmentIndex: number }
): { list: SegmentListKey; index: number } | null {
	if (target.segmentId) {
		const inPositive = lists.positive.findIndex((s) => s.id === target.segmentId);
		if (inPositive !== -1) return { list: 'positive', index: inPositive };
		const inNegative = lists.negative.findIndex((s) => s.id === target.segmentId);
		if (inNegative !== -1) return { list: 'negative', index: inNegative };
	}
	const flat = target.segmentIndex;
	if (flat >= 0 && flat < lists.positive.length) return { list: 'positive', index: flat };
	const negativeIndex = flat - lists.positive.length;
	if (flat >= 0 && negativeIndex < lists.negative.length) return { list: 'negative', index: negativeIndex };
	return null;
}

export async function applySegmentAdd(
	lists: SegmentLists,
	target: { segmentId: string; segmentIndex: number; content: string; list?: SegmentListKey },
	resourceContext?: ResourceTokenContext
): Promise<{ lists: SegmentLists; list: SegmentListKey; index: number } | null> {
	if (!target.content.trim()) return null;
	const anchor = locateSegmentInLists(lists, target);
	const list: SegmentListKey = target.list ?? anchor?.list ?? 'positive';
	const source = lists[list];
	const blank = createBlankEditorSegment();
	let seeded: Segment[];
	if (source.length === 1 && isPristineBlankSegment(source[0])) {
		seeded = [blank];
	} else if (anchor && anchor.list === list) {
		seeded = [...source.slice(0, anchor.index + 1), blank, ...source.slice(anchor.index + 1)];
	} else {
		seeded = [...source, blank];
	}
	const result = await applySegmentUpdate(
		seeded,
		{ segmentId: blank.id, segmentIndex: -1, content: target.content },
		resourceContext
	);
	if (!result) return null;
	return { lists: { ...lists, [list]: result.segments }, list, index: result.index };
}

export function applySegmentDelete(
	lists: SegmentLists,
	target: { segmentId: string; segmentIndex: number }
): { lists: SegmentLists; removal: SegmentRemoval } | null {
	const found = locateSegmentInLists(lists, target);
	if (!found) return null;
	const source = lists[found.list];
	const segment = source[found.index];
	const remaining = source.filter((_, i) => i !== found.index);
	return {
		lists: { ...lists, [found.list]: ensureSegmentList(remaining) },
		removal: {
			list: found.list,
			index: found.index,
			segment,
			previousId: source[found.index - 1]?.id ?? null,
			nextId: source[found.index + 1]?.id ?? null
		}
	};
}

export function restoreSegment(lists: SegmentLists, removal: SegmentRemoval): SegmentLists {
	const source = lists[removal.list];
	if (source.some((s) => s.id === removal.segment.id)) return lists;
	if (source.length === 1 && isPristineBlankSegment(source[0])) {
		return { ...lists, [removal.list]: [removal.segment] };
	}
	const previous = removal.previousId ? source.findIndex((s) => s.id === removal.previousId) : -1;
	const next = removal.nextId ? source.findIndex((s) => s.id === removal.nextId) : -1;
	const at = previous !== -1 ? previous + 1 : next !== -1 ? next : Math.min(removal.index, source.length);
	return {
		...lists,
		[removal.list]: [...source.slice(0, at), removal.segment, ...source.slice(at)]
	};
}
