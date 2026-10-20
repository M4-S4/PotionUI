import type { PromptTabData } from '$lib/types/tabs';
import type { ResourceTokenContext } from '$lib/utils/promptResourceTokens';
import { flattenRichSegments } from '$lib/utils/richSegments';
import {
	applySegmentAdd,
	applySegmentDelete,
	applySegmentUpdate,
	locateSegmentInLists,
	restoreSegment,
	type SegmentListKey,
	type SegmentLists,
	type SegmentRemoval
} from '$lib/utils/promptSegments';
import {
	MAX_BATCH_IMAGES,
	activePromptIndex,
	addPromptWith,
	deletePrompt,
	emptyPrompt,
	insertPrompt,
	promptCount,
	promptList,
	renamePrompt,
	replacePrompt,
	selectPrompt,
	withPromptIds,
	type PromptFields,
	type PromptListPatch
} from '$lib/generation/promptBatch';

export interface PromptListActionInput {
	type: string;
	promptIndex?: number;
	name?: string;
	content: string;
}

export interface PromptSegmentActionInput {
	type: string;
	segmentIndex: number;
	segmentId: string;
	content: string;
	list?: SegmentListKey;
}

export type PromptUndo =
	| { kind: 'add'; id: string; index: number; previousActiveId?: string }
	| { kind: 'remove'; index: number; entry: PromptTabData; previousActiveId?: string }
	| { kind: 'select'; previousActive: number; previousActiveId?: string }
	| { kind: 'rename'; id: string; previousName: string };

export interface PromptActionResult {
	patch: PromptListPatch;
	undo: PromptUndo;
}

export interface PromptSegmentResult {
	patch: PromptListPatch;
	segmentId: string;
	removal?: SegmentRemoval;
}

function listsOf(entry: PromptTabData): SegmentLists {
	return { positive: entry.promptSegments || [], negative: entry.negativePromptSegments || [] };
}

function entryWithLists(entry: PromptTabData, lists: SegmentLists): PromptTabData {
	return {
		...entry,
		promptSegments: lists.positive,
		negativePromptSegments: lists.negative,
		prompt: flattenRichSegments(lists.positive),
		negativePrompt: flattenRichSegments(lists.negative)
	};
}

export function isValidPromptIndex(tab: PromptFields, index: number | undefined): index is number {
	return index !== undefined && Number.isInteger(index) && index >= 0 && index < promptCount(tab);
}

function idAt(list: PromptTabData[] | undefined, index: number): string | undefined {
	return list?.[index]?.id;
}

function indexOfId(tab: PromptFields, id: string | undefined): number {
	if (!id || promptCount(tab) < 2) return -1;
	return tab.promptTabs!.findIndex((entry) => entry.id === id);
}

export async function applyPromptListAction(
	rawTab: PromptFields,
	action: PromptListActionInput,
	resourceContext?: ResourceTokenContext
): Promise<PromptActionResult | null> {
	const tab = withPromptIds(rawTab);
	const previousActive = activePromptIndex(tab);
	if (action.type === 'add_prompt') {
		const count = promptCount(tab);
		if (count >= MAX_BATCH_IMAGES) return null;
		let entry = emptyPrompt();
		if (action.name?.trim()) entry.name = action.name.trim();
		if (action.content.trim()) {
			const added = await applySegmentAdd(
				listsOf(entry),
				{ segmentId: '', segmentIndex: -1, content: action.content, list: 'positive' },
				resourceContext
			);
			if (added) entry = entryWithLists(entry, added.lists);
		}
		const patch = addPromptWith(tab, entry);
		return {
			patch,
			undo: { kind: 'add', id: entry.id!, index: count, previousActiveId: idAt(patch.promptTabs, previousActive) }
		};
	}
	if (!isValidPromptIndex(tab, action.promptIndex)) return null;
	const index = action.promptIndex;
	const previousActiveId = idAt(tab.promptTabs, previousActive);
	if (action.type === 'select_prompt') {
		return { patch: selectPrompt(tab, index), undo: { kind: 'select', previousActive, previousActiveId } };
	}
	if (action.type === 'remove_prompt') {
		const entry = promptList(tab)[index];
		const patch = deletePrompt(tab, index);
		if (!patch) return null;
		return { patch, undo: { kind: 'remove', index, entry, previousActiveId } };
	}
	if (action.type === 'rename_prompt') {
		const name = action.name?.trim();
		const id = idAt(tab.promptTabs, index);
		if (!name || !id) return null;
		const previousName = promptList(tab)[index]?.name ?? '';
		return { patch: renamePrompt(tab, index, name), undo: { kind: 'rename', id, previousName } };
	}
	return null;
}

export function undoPromptListAction(rawTab: PromptFields, undo: PromptUndo): PromptListPatch | null {
	const tab = withPromptIds(rawTab);
	if (undo.kind === 'add') {
		const index = indexOfId(tab, undo.id);
		if (index < 0) return null;
		const patch = deletePrompt(tab, index);
		if (!patch) return null;
		const restored = { ...tab, ...patch };
		const previous = indexOfId(restored, undo.previousActiveId);
		return previous >= 0 ? selectPrompt(restored, previous) : patch;
	}
	if (undo.kind === 'remove') {
		if (indexOfId(tab, undo.entry.id) >= 0) return null;
		const at = Math.min(undo.index, promptCount(tab));
		const withEntry = insertPrompt(tab, at, undo.entry, activePromptIndex(tab));
		const previous = indexOfId({ ...tab, ...withEntry }, undo.previousActiveId);
		return previous >= 0 ? insertPrompt(tab, at, undo.entry, previous) : withEntry;
	}
	if (undo.kind === 'select') {
		const byId = indexOfId(tab, undo.previousActiveId);
		if (byId >= 0) return selectPrompt(tab, byId);
		if (undo.previousActiveId) return null;
		return isValidPromptIndex(tab, undo.previousActive) ? selectPrompt(tab, undo.previousActive) : null;
	}
	const index = indexOfId(tab, undo.id);
	if (index < 0) return null;
	return renamePrompt(tab, index, undo.previousName);
}

export async function applySegmentActionToPrompt(
	tab: PromptFields,
	promptIndex: number,
	action: PromptSegmentActionInput,
	resourceContext?: ResourceTokenContext
): Promise<PromptSegmentResult | null> {
	if (!isValidPromptIndex(tab, promptIndex)) return null;
	const entry = promptList(tab)[promptIndex];
	const lists = listsOf(entry);
	let next: SegmentLists;
	let segmentId: string;
	let removal: SegmentRemoval | undefined;
	if (action.type === 'add_segment') {
		const result = await applySegmentAdd(lists, action, resourceContext);
		if (!result) return null;
		next = result.lists;
		segmentId = result.lists[result.list][result.index].id;
	} else if (action.type === 'delete_segment') {
		const result = applySegmentDelete(lists, action);
		if (!result) return null;
		next = result.lists;
		removal = { ...result.removal, prompt: promptIndex };
		segmentId = result.removal.segment.id;
	} else {
		const found = locateSegmentInLists(lists, action);
		if (!found) return null;
		const result = await applySegmentUpdate(
			lists[found.list],
			{ segmentId: action.segmentId, segmentIndex: found.index, content: action.content },
			resourceContext
		);
		if (!result) return null;
		next = { ...lists, [found.list]: result.segments };
		segmentId = result.segments[result.index].id;
	}
	const patch = replacePrompt(tab, promptIndex, entryWithLists(entry, next));
	return patch ? { patch, segmentId, removal } : null;
}

export function restoreSegmentInPrompt(
	tab: PromptFields,
	promptIndex: number,
	removal: SegmentRemoval
): PromptListPatch | null {
	if (!isValidPromptIndex(tab, promptIndex)) return null;
	const entry = promptList(tab)[promptIndex];
	return replacePrompt(tab, promptIndex, entryWithLists(entry, restoreSegment(listsOf(entry), removal)));
}

export function segmentTextInPrompt(
	tab: PromptFields,
	promptIndex: number,
	target: { segmentId: string; segmentIndex: number }
): string | null {
	if (!isValidPromptIndex(tab, promptIndex)) return null;
	const lists = listsOf(promptList(tab)[promptIndex]);
	const found = locateSegmentInLists(lists, target);
	return found ? (lists[found.list][found.index]?.content ?? null) : null;
}

export function promptActionHeading(action: PromptListActionInput, nextNumber = 0): string {
	const number = action.promptIndex !== undefined ? action.promptIndex + 1 : 0;
	if (action.type === 'add_prompt') {
		const label = nextNumber > 0 ? `Add prompt ${nextNumber}` : 'Add prompt';
		return action.name ? `${label} · ${action.name}` : label;
	}
	if (action.type === 'select_prompt') return `Switch to prompt ${number}`;
	if (action.type === 'remove_prompt') return `Remove prompt ${number}`;
	return `Rename prompt ${number}`;
}
