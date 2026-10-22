// @vitest-environment jsdom
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { ChipData } from '$lib/types/segments';
import { getCaretCharOffset, placeCaretAtCharOffset } from './chipEditorCaret';
import { extractContentFromDOM } from './chipEditorDom';

function makeChip(overrides: Partial<ChipData> = {}): ChipData {
	return {
		id: 'chip-1',
		categoryPath: 'cat', // encodes to "#cat" (4 chars)
		valueId: 'val-1',
		label: 'Cat',
		value: 'cat',
		allValues: [],
		shuffle: false,
		autoRegen: false,
		...overrides
	};
}

// hello <chip #cat, 4 chars> world  -- caret-relevant offsets:
// "hello " = 0..6, chip = 6..10, " world" = 10..16
function buildEditor(): { editor: HTMLDivElement; chip: HTMLElement; before: Text; after: Text } {
	const editor = document.createElement('div');
	const chipData = makeChip();
	const before = document.createTextNode('hello ');
	const chip = document.createElement('span');
	chip.dataset.chipId = chipData.id;
	chip.contentEditable = 'false';
	const after = document.createTextNode(' world');
	editor.appendChild(before);
	editor.appendChild(chip);
	editor.appendChild(after);
	document.body.appendChild(editor);
	return { editor, chip, before, after };
}

const chips: Record<string, ChipData> = { 'chip-1': makeChip() };

let cleanup: HTMLElement[] = [];
beforeEach(() => {
	cleanup = [];
});
afterEach(() => {
	cleanup.forEach((el) => el.remove());
	window.getSelection()?.removeAllRanges();
});

function withCleanup<T extends { editor: HTMLElement }>(built: T): T {
	cleanup.push(built.editor);
	return built;
}

// <chip1 #cat, 4 chars><chip2 #dog, 4 chars> -- no text nodes at all:
// chip1 = 0..4, chip2 = 4..8, end of content = 8
function buildAdjacentChipsEditor(): { editor: HTMLDivElement; chip1: HTMLElement; chip2: HTMLElement } {
	const editor = document.createElement('div');
	const chip1 = document.createElement('span');
	chip1.dataset.chipId = 'chip-1';
	chip1.contentEditable = 'false';
	const chip2 = document.createElement('span');
	chip2.dataset.chipId = 'chip-2';
	chip2.contentEditable = 'false';
	editor.appendChild(chip1);
	editor.appendChild(chip2);
	document.body.appendChild(editor);
	return { editor, chip1, chip2 };
}

const adjacentChips: Record<string, ChipData> = {
	'chip-1': makeChip({ id: 'chip-1', categoryPath: 'cat' }),
	'chip-2': makeChip({ id: 'chip-2', categoryPath: 'dog' })
};

describe('getCaretCharOffset', () => {
	it('returns null when there is no active selection range', () => {
		const { editor } = withCleanup(buildEditor());
		window.getSelection()?.removeAllRanges();
		expect(getCaretCharOffset(editor, chips)).toBeNull();
	});

	it('returns null when editorRef is null/undefined', () => {
		expect(getCaretCharOffset(null, chips)).toBeNull();
		expect(getCaretCharOffset(undefined, chips)).toBeNull();
	});

	it('returns null for a non-collapsed (range) selection', () => {
		const { editor, before, after } = withCleanup(buildEditor());
		const range = document.createRange();
		range.setStart(before, 0);
		range.setEnd(after, 2);
		const sel = window.getSelection()!;
		sel.removeAllRanges();
		sel.addRange(range);
		expect(getCaretCharOffset(editor, chips)).toBeNull();
	});

	it('measures a given range instead of the live selection', () => {
		const { editor, after } = withCleanup(buildEditor());
		window.getSelection()?.removeAllRanges();
		const range = document.createRange();
		range.setStart(after, 2);
		range.collapse(true);
		expect(getCaretCharOffset(editor, chips, range)).toBe(12);
	});

	it('computes the absolute offset for a caret in the text node before the chip', () => {
		const { editor, before } = withCleanup(buildEditor());
		const range = document.createRange();
		range.setStart(before, 3);
		range.collapse(true);
		const sel = window.getSelection()!;
		sel.removeAllRanges();
		sel.addRange(range);
		expect(getCaretCharOffset(editor, chips)).toBe(3);
	});

	it('computes the absolute offset for a caret in the text node after the chip, counting the chip as its encoded #path length', () => {
		const { editor, after } = withCleanup(buildEditor());
		const range = document.createRange();
		range.setStart(after, 2); // " w|orld"
		range.collapse(true);
		const sel = window.getSelection()!;
		sel.removeAllRanges();
		sel.addRange(range);
		// "hello " (6) + "#cat" (4) + 2
		expect(getCaretCharOffset(editor, chips)).toBe(12);
	});

	// When the caret sits immediately after an atomic container as produced by
	// Range.setStartAfter, startContainer === editorRef itself (per the DOM
	// spec, not a jsdom quirk) and startOffset is a child index - not a
	// same-node child of editorRef, which is all the recursive walk below
	// compares against.
	it('resolves the offset when the caret is placed via setStartAfter on the chip container (startContainer === editorRef)', () => {
		const { editor, chip } = withCleanup(buildEditor());
		const range = document.createRange();
		range.setStartAfter(chip);
		range.collapse(true);
		expect(range.startContainer).toBe(editor); // per DOM spec, not jsdom-specific
		const sel = window.getSelection()!;
		sel.removeAllRanges();
		sel.addRange(range);
		// "hello " (6) + "#cat" (4)
		expect(getCaretCharOffset(editor, chips)).toBe(10);
	});

	it('resolves the offset when the caret sits between two adjacent chips', () => {
		const { editor, chip1 } = withCleanup(buildAdjacentChipsEditor());
		const range = document.createRange();
		range.setStartAfter(chip1);
		range.collapse(true);
		expect(range.startContainer).toBe(editor);
		const sel = window.getSelection()!;
		sel.removeAllRanges();
		sel.addRange(range);
		expect(getCaretCharOffset(editor, adjacentChips)).toBe(4);
	});

	it('resolves the offset when the caret sits after the last chip with no trailing text node', () => {
		const { editor, chip2 } = withCleanup(buildAdjacentChipsEditor());
		const range = document.createRange();
		range.setStartAfter(chip2);
		range.collapse(true);
		expect(range.startContainer).toBe(editor);
		const sel = window.getSelection()!;
		sel.removeAllRanges();
		sel.addRange(range);
		expect(getCaretCharOffset(editor, adjacentChips)).toBe(8);
	});
});

describe('placeCaretAtCharOffset', () => {
	it('places the caret inside a text node at the given offset', () => {
		const { editor, before } = withCleanup(buildEditor());
		placeCaretAtCharOffset(editor, chips, 3);
		const range = window.getSelection()!.getRangeAt(0);
		expect(range.startContainer).toBe(before);
		expect(range.startOffset).toBe(3);
		expect(range.collapsed).toBe(true);
	});

	it('lands just AFTER the chip container when the target offset is exactly at its trailing boundary', () => {
		const { editor, chip } = withCleanup(buildEditor());
		placeCaretAtCharOffset(editor, chips, 10); // 6 ("hello ") + 4 ("#cat")
		const range = window.getSelection()!.getRangeAt(0);
		// setStartAfter semantics: anchors on the parent, offset = index+1.
		expect(range.startContainer).toBe(editor);
		expect(range.startOffset).toBe(Array.from(editor.childNodes).indexOf(chip) + 1);
	});

	it('places the caret in the text node after the chip when the offset is past the boundary', () => {
		const { editor, after } = withCleanup(buildEditor());
		placeCaretAtCharOffset(editor, chips, 12); // 2 chars into " world"
		const range = window.getSelection()!.getRangeAt(0);
		expect(range.startContainer).toBe(after);
		expect(range.startOffset).toBe(2);
	});

	it('falls back to the end of editorRef content when the target offset exceeds the total length', () => {
		const { editor } = withCleanup(buildEditor());
		placeCaretAtCharOffset(editor, chips, 9999);
		const range = window.getSelection()!.getRangeAt(0);
		expect(range.collapsed).toBe(true);
		// selectNodeContents(editorRef) + collapse(false) anchors on editorRef
		// at its full child count.
		expect(range.startContainer).toBe(editor);
		expect(range.startOffset).toBe(editor.childNodes.length);
	});

	it('is a no-op when editorRef is null/undefined (no throw)', () => {
		expect(() => placeCaretAtCharOffset(null, chips, 3)).not.toThrow();
		expect(() => placeCaretAtCharOffset(undefined, chips, 3)).not.toThrow();
	});

	it('round-trips with getCaretCharOffset for an offset strictly inside a text node', () => {
		const { editor } = withCleanup(buildEditor());
		placeCaretAtCharOffset(editor, chips, 3);
		expect(getCaretCharOffset(editor, chips)).toBe(3);
	});

	it('round-trips with getCaretCharOffset for an offset in the text node past the chip', () => {
		const { editor } = withCleanup(buildEditor());
		placeCaretAtCharOffset(editor, chips, 12);
		expect(getCaretCharOffset(editor, chips)).toBe(12);
	});

	it('round-trips with getCaretCharOffset when landing exactly at a chip boundary', () => {
		const { editor } = withCleanup(buildEditor());
		placeCaretAtCharOffset(editor, chips, 10);
		expect(getCaretCharOffset(editor, chips)).toBe(10);
	});

	it('round-trips with getCaretCharOffset when landing between two adjacent chips', () => {
		const { editor } = withCleanup(buildAdjacentChipsEditor());
		placeCaretAtCharOffset(editor, adjacentChips, 4);
		expect(getCaretCharOffset(editor, adjacentChips)).toBe(4);
	});

	it('round-trips with getCaretCharOffset when landing after the last chip with no trailing text node', () => {
		const { editor } = withCleanup(buildAdjacentChipsEditor());
		placeCaretAtCharOffset(editor, adjacentChips, 8);
		expect(getCaretCharOffset(editor, adjacentChips)).toBe(8);
	});
});

function buildLines(lines: string[]): { editor: HTMLDivElement; texts: Text[] } {
	const editor = document.createElement('div');
	const texts = lines.map((line) => {
		const block = document.createElement('div');
		const text = document.createTextNode(line);
		block.appendChild(text);
		editor.appendChild(block);
		return text;
	});
	document.body.appendChild(editor);
	return { editor, texts };
}

function setCaret(node: Node, offset: number) {
	const range = document.createRange();
	range.setStart(node, offset);
	range.collapse(true);
	const selection = window.getSelection()!;
	selection.removeAllRanges();
	selection.addRange(range);
}

describe('offsets across Enter-created blocks', () => {
	const lines = ['line1', 'line2', 'line3<<'];

	it('measures a caret on each line against the extracted value', () => {
		const { editor, texts } = withCleanup(buildLines(lines));
		const { value } = extractContentFromDOM(editor, {});
		expect(value).toBe('line1\nline2\nline3<<');
		lines.forEach((line, index) => {
			setCaret(texts[index], line.length);
			expect(getCaretCharOffset(editor, {})).toBe(lines.slice(0, index + 1).join('\n').length);
		});
		setCaret(texts[2], 7);
		const caret = getCaretCharOffset(editor, {})!;
		expect(value.slice(caret - 2, caret)).toBe('<<');
	});

	it('counts the breaks before a caret anchored on the editor element', () => {
		const { editor } = withCleanup(buildLines(lines));
		setCaret(editor, 2);
		expect(getCaretCharOffset(editor, {})).toBe('line1\nline2\n'.length);
		setCaret(editor, 3);
		expect(getCaretCharOffset(editor, {})).toBe('line1\nline2\nline3<<'.length);
	});

	it('places the caret on the line the offset belongs to', () => {
		const { editor, texts } = withCleanup(buildLines(lines));
		placeCaretAtCharOffset(editor, {}, 5);
		expect(window.getSelection()!.getRangeAt(0).startContainer).toBe(texts[0]);
		expect(window.getSelection()!.getRangeAt(0).startOffset).toBe(5);
		placeCaretAtCharOffset(editor, {}, 6);
		expect(window.getSelection()!.getRangeAt(0).startContainer).toBe(texts[1]);
		expect(window.getSelection()!.getRangeAt(0).startOffset).toBe(0);
		placeCaretAtCharOffset(editor, {}, 15);
		expect(window.getSelection()!.getRangeAt(0).startContainer).toBe(texts[2]);
		expect(window.getSelection()!.getRangeAt(0).startOffset).toBe(3);
	});

	it('round-trips every offset of the extracted value', () => {
		const { editor } = withCleanup(buildLines(lines));
		const { value } = extractContentFromDOM(editor, {});
		for (let offset = 0; offset <= value.length; offset++) {
			placeCaretAtCharOffset(editor, {}, offset);
			expect(getCaretCharOffset(editor, {})).toBe(offset);
		}
	});

	it('counts a chip after the breaks at its place in the value', () => {
		const { editor, texts } = withCleanup(buildLines(['line1', 'line2']));
		const chip = document.createElement('span');
		chip.dataset.chipId = 'chip-1';
		chip.contentEditable = 'false';
		editor.appendChild(chip);
		const { value } = extractContentFromDOM(editor, chips);
		expect(value).toBe('line1\nline2\n#cat');
		setCaret(texts[1], 5);
		expect(getCaretCharOffset(editor, chips)).toBe(11);
		placeCaretAtCharOffset(editor, chips, value.length);
		const range = window.getSelection()!.getRangeAt(0);
		expect(range.startContainer).toBe(editor);
		expect(range.startOffset).toBe(3);
		expect(getCaretCharOffset(editor, chips)).toBe(value.length);
	});
});

describe('a caret anchored on an Enter-created block element', () => {
	function buildChipLines() {
		const editor = document.createElement('div');
		const first = document.createElement('div');
		const firstText = document.createTextNode('a');
		const firstChip = document.createElement('span');
		firstChip.dataset.chipId = 'chip-1';
		firstChip.contentEditable = 'false';
		first.append(firstText, firstChip);
		const second = document.createElement('div');
		const secondText = document.createTextNode('line2');
		const secondChip = document.createElement('span');
		secondChip.dataset.chipId = 'chip-1';
		secondChip.contentEditable = 'false';
		second.append(secondText, secondChip);
		editor.append(first, second);
		document.body.appendChild(editor);
		return { editor, first, second, firstChip, secondChip };
	}

	it('counts the children before the anchor as characters, chips by their encoded path', () => {
		const { editor, first, second } = withCleanup(buildChipLines());
		const { value } = extractContentFromDOM(editor, chips);
		expect(value).toBe('a#cat\nline2#cat');
		setCaret(first, 1);
		expect(getCaretCharOffset(editor, chips)).toBe(1);
		setCaret(first, 2);
		expect(getCaretCharOffset(editor, chips)).toBe(5);
		setCaret(second, 1);
		expect(getCaretCharOffset(editor, chips)).toBe(11);
		setCaret(second, 2);
		expect(getCaretCharOffset(editor, chips)).toBe(value.length);
	});

	it('round-trips the caret that placeCaretAtCharOffset puts right after a chip in a block', () => {
		const { editor, first, second } = withCleanup(buildChipLines());
		placeCaretAtCharOffset(editor, chips, 5);
		let range = window.getSelection()!.getRangeAt(0);
		expect(range.startContainer).toBe(first);
		expect(range.startOffset).toBe(2);
		expect(getCaretCharOffset(editor, chips)).toBe(5);
		placeCaretAtCharOffset(editor, chips, 15);
		range = window.getSelection()!.getRangeAt(0);
		expect(range.startContainer).toBe(second);
		expect(range.startOffset).toBe(2);
		expect(getCaretCharOffset(editor, chips)).toBe(15);
	});

	it('round-trips every offset outside the chips', () => {
		const { editor } = withCleanup(buildChipLines());
		const outside = [0, 1, 5, 6, 7, 8, 9, 10, 11, 15];
		outside.forEach((offset) => {
			placeCaretAtCharOffset(editor, chips, offset);
			expect(getCaretCharOffset(editor, chips)).toBe(offset);
		});
	});
});

