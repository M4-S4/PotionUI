// Caret <-> absolute-char-offset conversion for InlineChipEditor.svelte,
// extracted unchanged. Reads/writes `window.getSelection()` directly,
// matching the original component methods.
import type { ChipData } from '$lib/types/segments';
import { encodePathForText } from './chipSegments';
import { breakElementLength, endsWithLineBreak, startsWithLineBreak } from './chipEditorDom';

/** Per-node contribution to the absolute char offset, matching how
 * `extractContentFromDOM` linearizes the editor's content. Shared by the
 * `startContainer === editorRef` branch below and (indirectly, by the same
 * accounting rules) the recursive walk that follows it. */
function nodeCharLength(node: Node, chips: Record<string, ChipData>): number {
	if (node.nodeType === Node.TEXT_NODE) {
		return (node.textContent || '').length;
	}
	if (node.nodeType !== Node.ELEMENT_NODE) return 0;
	const el = node as HTMLElement;
	if (el.dataset.groupRaw !== undefined) return el.dataset.groupRaw.length;
	if (el.dataset.variableRaw !== undefined) return el.dataset.variableRaw.length;
	if (el.dataset.resourceMarker !== undefined) return el.dataset.resourceMarker.length;
	if (el.dataset.chipId && chips[el.dataset.chipId]) {
		return encodePathForText(chips[el.dataset.chipId].categoryPath).length;
	}
	if (el.tagName === 'BR') return breakElementLength(el);
	let sum = startsWithLineBreak(el) ? 1 : 0;
	for (const child of Array.from(node.childNodes)) sum += nodeCharLength(child, chips);
	return endsWithLineBreak(el) ? sum + 1 : sum;
}

function boundaryCharLength(container: Node, offset: number, chips: Record<string, ChipData>): number {
	if (container.nodeType === Node.TEXT_NODE) return offset;
	let sum = 0;
	for (const child of Array.from(container.childNodes).slice(0, offset)) sum += nodeCharLength(child, chips);
	return sum;
}

/** Absolute character offset of the caret in the text `extractContentFromDOM` would produce, or null if unavailable. */
export function getCaretCharOffset(
	editorRef: HTMLElement | undefined | null,
	chips: Record<string, ChipData>,
	at?: Range
): number | null {
	const selection = window.getSelection();
	const current = at ?? (selection?.rangeCount ? selection.getRangeAt(0) : null);
	if (!current || !editorRef || !current.collapsed) return null;
	const range = current;

	// Range.setStartAfter() (what placeCaretAtCharOffset uses to land "just
	// after" a chip/group/variable, and what the browser itself produces for
	// an arrow-key move off an atomic contentEditable=false container) anchors
	// directly on editorRef with startOffset as a CHILD INDEX, per the DOM
	// spec - not something the recursive walk below (which only ever compares
	// editorRef's descendants, never editorRef itself) can match. A collapsed
	// range's end boundary equals its start boundary by definition, so
	// endContainer/endOffset never need separate handling here.
	if (range.startContainer === editorRef) {
		return boundaryCharLength(editorRef, range.startOffset, chips);
	}

	let offset = 0;
	let found = false;

	function walk(node: Node): boolean {
		if (found) return true;
		if (node === range.startContainer) {
			if (node.nodeType === Node.ELEMENT_NODE && startsWithLineBreak(node as HTMLElement)) offset += 1;
			offset += boundaryCharLength(node, range.startOffset, chips);
			found = true;
			return true;
		}
		if (node.nodeType === Node.TEXT_NODE) {
			offset += (node.textContent || '').length;
		} else if (node.nodeType === Node.ELEMENT_NODE) {
			const el = node as HTMLElement;
			if (el.dataset.groupRaw !== undefined) {
				offset += el.dataset.groupRaw.length;
			} else if (el.dataset.variableRaw !== undefined) {
				offset += el.dataset.variableRaw.length;
			} else if (el.dataset.resourceMarker !== undefined) {
				offset += el.dataset.resourceMarker.length;
			} else if (el.dataset.chipId && chips[el.dataset.chipId]) {
				offset += encodePathForText(chips[el.dataset.chipId].categoryPath).length;
			} else if (el.tagName === 'BR') {
				offset += breakElementLength(el);
			} else {
				if (startsWithLineBreak(el)) offset += 1;
				for (const child of Array.from(node.childNodes)) {
					if (walk(child)) return true;
				}
				if (endsWithLineBreak(el)) offset += 1;
			}
		}
		return false;
	}

	for (const child of Array.from(editorRef.childNodes)) {
		if (walk(child)) break;
	}

	return found ? offset : null;
}

export function triggerCharOffset(
	editorRef: HTMLElement | undefined | null,
	chips: Record<string, ChipData>,
	node: Text | null,
	offset: number
): number | null {
	if (!editorRef || !node || offset < 0) return null;
	const range = document.createRange();
	range.setStart(node, Math.min(offset, node.length));
	range.collapse(true);
	return getCaretCharOffset(editorRef, chips, range);
}

/** Inverse of getCaretCharOffset: place the caret at an absolute char offset, landing just after an atomic (chip/group) container if the offset falls at its boundary. */
export function placeCaretAtCharOffset(
	editorRef: HTMLElement | undefined | null,
	chips: Record<string, ChipData>,
	target: number
) {
	if (!editorRef) return;
	const selection = window.getSelection();
	if (!selection) return;

	let remaining = target;

	function walk(node: Node): { node: Node; afterNode: boolean } | null {
		if (node.nodeType === Node.TEXT_NODE) {
			const len = (node.textContent || '').length;
			if (remaining <= len) return { node, afterNode: false };
			remaining -= len;
			return null;
		}
		if (node.nodeType === Node.ELEMENT_NODE) {
			const el = node as HTMLElement;
			if (el.dataset.groupRaw !== undefined) {
				const len = el.dataset.groupRaw.length;
				if (remaining <= len) return { node: el, afterNode: true };
				remaining -= len;
				return null;
			}
			if (el.dataset.variableRaw !== undefined) {
				const len = el.dataset.variableRaw.length;
				if (remaining <= len) return { node: el, afterNode: true };
				remaining -= len;
				return null;
			}
			if (el.dataset.resourceMarker !== undefined) {
				const len = el.dataset.resourceMarker.length;
				if (remaining <= len) return { node: el, afterNode: true };
				remaining -= len;
				return null;
			}
			if (el.dataset.chipId) {
				const len = encodePathForText(chips[el.dataset.chipId]?.categoryPath || '').length;
				if (remaining <= len) return { node: el, afterNode: true };
				remaining -= len;
				return null;
			}
			if (el.tagName === 'BR') {
				if (breakElementLength(el) === 0) {
					return remaining <= 0 ? { node: el.parentNode as Node, afterNode: false } : null;
				}
				if (remaining <= 1) return { node: el, afterNode: true };
				remaining -= 1;
				return null;
			}
			if (startsWithLineBreak(el)) remaining -= 1;
			for (const child of Array.from(node.childNodes)) {
				const result = walk(child);
				if (result) return result;
			}
			if (endsWithLineBreak(el)) remaining -= 1;
		}
		return null;
	}

	let result: { node: Node; afterNode: boolean } | null = null;
	for (const child of Array.from(editorRef.childNodes)) {
		result = walk(child);
		if (result) break;
	}

	const range = document.createRange();
	if (result) {
		if (result.afterNode) {
			range.setStartAfter(result.node);
		} else {
			range.setStart(result.node, Math.max(0, remaining));
		}
	} else {
		range.selectNodeContents(editorRef);
		range.collapse(false);
	}
	range.collapse(true);
	selection.removeAllRanges();
	selection.addRange(range);
}
