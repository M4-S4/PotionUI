import { describe, it, expect, afterEach, vi } from 'vitest';
import { flushSync, tick } from 'svelte';
import type { VariablesMap } from '$lib/utils/variableDefs';
import { getCaretCharOffset } from '$lib/components/chipEditorCaret';
import { registerDismissLayer, resetLayerStackForTests } from '$lib/actions/layerStack';
import dismiss from '$lib/actions/dismiss';

const { default: InlineChipEditor } = await import('$lib/components/InlineChipEditor.svelte');
const { createClassComponent } = await import('svelte/legacy');

const variables: VariablesMap = {
	x: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['calm', 'tense'] }
};
const DEFAULT = '<<if $x = "calm">>text<<end>>';
const STORM = '<<if $x = "calm">>storm<<end>>';

let target: HTMLDivElement;
let component: ReturnType<typeof createClassComponent> | undefined;

function mountEditor(value: string, props: Record<string, unknown> = {}, host: HTMLElement = document.body) {
	target = document.createElement('div');
	host.appendChild(target);
	component = createClassComponent({
		component: InlineChipEditor as never,
		target,
		props: { value, chips: {}, variant: 'segment-composer', borderless: true, variables, ...props }
	});
	const values: string[] = [];
	const handle = component as unknown as {
		$on: (n: string, cb: (e: CustomEvent) => void) => void;
		$set: (p: Record<string, unknown>) => void;
	};
	handle.$on('change', (e) => {
		values.push(e.detail.value);
		handle.$set({ value: e.detail.value });
	});
	flushSync();
	return {
		values,
		handle: handle as unknown as { insertCondition: () => void; $set: (p: Record<string, unknown>) => void },
		editor: target.querySelector<HTMLElement>('.inline-chip-editor')!
	};
}

afterEach(() => {
	component?.$destroy();
	component = undefined;
	target?.remove();
	document.body.innerHTML = '';
	resetLayerStackForTests();
});

async function settle() {
	flushSync();
	await tick();
	await tick();
}

function setCaret(node: Node, offset: number) {
	const range = document.createRange();
	range.setStart(node, offset);
	range.collapse(true);
	const selection = window.getSelection()!;
	selection.removeAllRanges();
	selection.addRange(range);
	document.dispatchEvent(new Event('selectionchange'));
}

function typeChar(editor: HTMLElement, character: string, init: InputEventInit = {}, origin: HTMLElement = editor) {
	const selection = window.getSelection()!;
	const range = selection.getRangeAt(0);
	let node = range.startContainer;
	let offset = range.startOffset;
	if (node.nodeType !== Node.TEXT_NODE) {
		const fresh = document.createTextNode('');
		node.insertBefore(fresh, node.childNodes[offset] ?? null);
		node = fresh;
		offset = 0;
	}
	const text = node as Text;
	text.data = text.data.slice(0, offset) + character + text.data.slice(offset);
	setCaret(text, offset + character.length);
	origin.dispatchEvent(
		new InputEvent('input', { bubbles: true, inputType: 'insertText', data: character, isComposing: false, ...init })
	);
}

const modal = () => document.querySelector<HTMLElement>('[role="dialog"][aria-labelledby="condition-modal-title"]');

const caretOffset = (editor: HTMLElement) => getCaretCharOffset(editor, {});

function caretIsAfter(editor: HTMLElement, element: Element) {
	const range = window.getSelection()!.getRangeAt(0);
	return range.startContainer === editor && editor.childNodes[range.startOffset - 1] === element;
}

async function typePair(editor: HTMLElement) {
	typeChar(editor, '<');
	await settle();
	typeChar(editor, '<');
	await settle();
}

async function saveModalWithText(text: string) {
	const dialog = modal()!;
	const area = dialog.querySelector<HTMLTextAreaElement>('textarea[aria-label="If text"]')!;
	area.value = text;
	area.dispatchEvent(new Event('input', { bubbles: true }));
	await tick();
	Array.from(dialog.querySelectorAll('button'))
		.find((b) => b.textContent?.trim().startsWith('Save'))!
		.click();
	await settle();
}

async function cancelModal() {
	Array.from(modal()!.querySelectorAll('button'))
		.find((b) => b.textContent?.trim().startsWith('Cancel'))!
		.click();
	await settle();
}

function startAfter(editor: HTMLElement, caret: number) {
	setCaret(editor.firstChild!, caret);
}

function toLines(editor: HTMLElement, lines: string[]) {
	editor.replaceChildren(
		...lines.map((line) => {
			const block = document.createElement('div');
			block.appendChild(document.createTextNode(line));
			return block;
		})
	);
	return Array.from(editor.children).map((block) => block.firstChild as Text);
}

describe('typing << in the prompt editor', () => {
	it('opens the condition editor on the second <, not the first', async () => {
		const { editor } = mountEditor('a cat');
		await settle();
		startAfter(editor, 5);
		typeChar(editor, '<');
		await settle();
		expect(modal()).toBeNull();
		typeChar(editor, '<');
		await settle();
		expect(modal()).not.toBeNull();
		expect(editor.textContent).toBe('a cat<<');
	});

	it('replaces the two characters with the condition at the caret in the middle of the text', async () => {
		const { editor, values } = mountEditor('a cat sits');
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		expect(values.at(-1)).toBe('a cat<< sits');
		await saveModalWithText('storm');
		expect(modal()).toBeNull();
		expect(values.at(-1)).toBe(`a cat${STORM} sits`);
		const container = editor.querySelector('.choice-group-container')!;
		expect(container.getAttribute('data-group-raw')).toBe(STORM);
		expect(editor.querySelectorAll('.condition-chip')).toHaveLength(1);
		expect(editor.textContent).not.toContain('<<');
		expect(caretIsAfter(editor, container)).toBe(true);
		expect(caretOffset(editor)).toBe(5 + STORM.length);
	});

	it('replaces them at the end of the text and leaves the caret after the chip', async () => {
		const { editor, values } = mountEditor('a cat');
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		await saveModalWithText('storm');
		expect(values.at(-1)).toBe(`a cat${STORM}`);
		expect(caretIsAfter(editor, editor.querySelector('.choice-group-container')!)).toBe(true);
		expect(caretOffset(editor)).toBe(5 + STORM.length);
	});

	it('replaces them at the start of the text', async () => {
		const { editor, values } = mountEditor(' sits');
		await settle();
		startAfter(editor, 0);
		await typePair(editor);
		await saveModalWithText('storm');
		expect(values.at(-1)).toBe(`${STORM} sits`);
		expect(caretOffset(editor)).toBe(STORM.length);
	});

	it('replaces them right after a variable chip', async () => {
		const { editor, values } = mountEditor('portrait of ${x}, soft light');
		await settle();
		const chip = editor.querySelector('.variable-usage-container')!;
		setCaret(chip.nextSibling!, 0);
		await typePair(editor);
		expect(values.at(-1)).toBe('portrait of ${x}<<, soft light');
		await saveModalWithText('storm');
		expect(values.at(-1)).toBe(`portrait of \${x}${STORM}, soft light`);
		expect(editor.querySelectorAll('.variable-usage-container')).toHaveLength(1);
		expect(caretIsAfter(editor, editor.querySelector('.choice-group-container')!)).toBe(true);
		expect(caretOffset(editor)).toBe('portrait of ${x}'.length + STORM.length);
	});

	it('inserts the saved condition once, with the default branch text untouched when nothing is edited', async () => {
		const { editor, values } = mountEditor('a cat sits');
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		Array.from(modal()!.querySelectorAll('button'))
			.find((b) => b.textContent?.trim().startsWith('Save'))!
			.click();
		await settle();
		expect(values.at(-1)).toBe(`a cat${DEFAULT} sits`);
		expect(editor.querySelectorAll('.condition-chip')).toHaveLength(1);
	});

	it('keeps << as literal text with the caret after it when the editor is cancelled', async () => {
		const { editor, values } = mountEditor('a cat sits');
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		await cancelModal();
		expect(modal()).toBeNull();
		expect(values.at(-1)).toBe('a cat<< sits');
		expect(editor.querySelector('.condition-chip')).toBeNull();
		expect(editor.textContent).toBe('a cat<< sits');
		expect(caretOffset(editor)).toBe(7);
		expect(document.activeElement).toBe(editor);
	});

	it('keeps << as literal text with the caret after it on Escape', async () => {
		const { editor, values } = mountEditor('a cat sits');
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
		await settle();
		expect(modal()).toBeNull();
		expect(values.at(-1)).toBe('a cat<< sits');
		expect(caretOffset(editor)).toBe(7);
		expect(document.activeElement).toBe(editor);
	});

	it('does not open again for a third or fourth <', async () => {
		const { editor, values } = mountEditor('a cat sits');
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		await cancelModal();
		typeChar(editor, '<');
		await settle();
		expect(modal()).toBeNull();
		typeChar(editor, '<');
		await settle();
		expect(modal()).toBeNull();
		expect(values.at(-1)).toBe('a cat<<<< sits');
	});

	it('does not open when a < lands between two existing ones', async () => {
		const { editor } = mountEditor('a cat<< sits');
		await settle();
		startAfter(editor, 6);
		typeChar(editor, '<');
		await settle();
		expect(modal()).toBeNull();
	});

	it('opens again once the run of < has been broken and retyped', async () => {
		const { editor } = mountEditor('a cat sits');
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		await cancelModal();
		const text = editor.firstChild as Text;
		text.data = 'a cat< sits';
		setCaret(text, 6);
		typeChar(editor, '<');
		await settle();
		expect(modal()).not.toBeNull();
	});
});

describe('typing << where it must not open the condition editor', () => {
	it('stays closed inside a chip', async () => {
		const { editor } = mountEditor('portrait of ${x}, soft light');
		await settle();
		const chip = editor.querySelector('.variable-usage-container')!;
		const inner = document.createTextNode('<');
		chip.appendChild(inner);
		setCaret(inner, 1);
		typeChar(editor, '<');
		await settle();
		expect(modal()).toBeNull();
	});

	it('stays closed when the caret sits between nodes rather than inside text', async () => {
		const { editor } = mountEditor('a cat sits');
		await settle();
		editor.replaceChildren(document.createTextNode('<<x'), document.createTextNode('yy'));
		setCaret(editor, 2);
		editor.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: '<' }));
		await settle();
		expect(modal()).toBeNull();
	});

	it('stays closed for an input raised from inside a chip popover', async () => {
		const { editor } = mountEditor('a cat sits');
		await settle();
		const field = document.createElement('input');
		editor.appendChild(field);
		startAfter(editor, 5);
		typeChar(editor, '<', {}, field);
		typeChar(editor, '<', {}, field);
		await settle();
		expect(modal()).toBeNull();
	});

	it('stays closed while another dialog is open', async () => {
		const { editor } = mountEditor('a cat sits');
		await settle();
		const other = document.createElement('div');
		document.body.appendChild(other);
		registerDismissLayer({ node: other });
		startAfter(editor, 5);
		await typePair(editor);
		expect(modal()).toBeNull();
	});

	it('stays closed when a popover sits above the dialog the editor is in', async () => {
		const host = document.createElement('div');
		document.body.appendChild(host);
		dismiss(host, { onDismiss: () => undefined });
		const { editor } = mountEditor('a cat sits', {}, host);
		await settle();
		const popover = document.createElement('div');
		document.body.appendChild(popover);
		registerDismissLayer({ node: popover });
		startAfter(editor, 5);
		await typePair(editor);
		expect(modal()).toBeNull();
	});

	it('stays closed while a condition chip editor is open', async () => {
		const { editor } = mountEditor(`a cat ${DEFAULT} sits`);
		await settle();
		editor.querySelector('.condition-chip .chip-main')!.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
		await settle();
		expect(document.querySelectorAll('[role="dialog"][aria-labelledby="condition-modal-title"]')).toHaveLength(1);
		const before = editor.firstChild as Text;
		setCaret(before, 3);
		await typePair(editor);
		expect(document.querySelectorAll('[role="dialog"][aria-labelledby="condition-modal-title"]')).toHaveLength(1);
	});

	it('stays closed during IME composition', async () => {
		const { editor } = mountEditor('a cat sits');
		await settle();
		startAfter(editor, 5);
		typeChar(editor, '<', { isComposing: true });
		typeChar(editor, '<', { isComposing: true });
		await settle();
		expect(editor.textContent).toBe('a cat<< sits');
		expect(modal()).toBeNull();
	});

	it('stays closed for composition text that is not flagged as composing', async () => {
		const { editor } = mountEditor('a cat< sits');
		await settle();
		startAfter(editor, 6);
		typeChar(editor, '<', { inputType: 'insertCompositionText', isComposing: false });
		await settle();
		expect(editor.textContent).toBe('a cat<< sits');
		expect(modal()).toBeNull();
	});

	it('stays closed when a single < is pasted after a <', async () => {
		const { editor } = mountEditor('a cat< sits');
		await settle();
		startAfter(editor, 6);
		const paste = new Event('paste', { bubbles: true, cancelable: true });
		Object.defineProperty(paste, 'clipboardData', { value: { getData: () => '<' } });
		editor.dispatchEvent(paste);
		await settle();
		expect(editor.textContent).toBe('a cat<< sits');
		expect(modal()).toBeNull();
	});

	it('stays closed for a pasted or multi-character input event', async () => {
		const { editor } = mountEditor('a cat sits');
		await settle();
		startAfter(editor, 5);
		typeChar(editor, '<<', { data: '<<' });
		await settle();
		expect(modal()).toBeNull();
		const text = editor.firstChild as Text;
		text.data = 'a cat< sits';
		setCaret(text, 6);
		typeChar(editor, '<', { inputType: 'insertFromPaste' });
		await settle();
		expect(modal()).toBeNull();
	});

	it('stays closed where conditions are not available', async () => {
		const { editor } = mountEditor('a cat sits', { variables: {} });
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		expect(modal()).toBeNull();
		expect(editor.textContent).toBe('a cat<< sits');
	});

	it('stays closed in a disabled segment', async () => {
		const { editor } = mountEditor('a cat sits', { segmentDisabled: true });
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		expect(modal()).toBeNull();
	});

	it('opens inside a dialog the editor itself sits in', async () => {
		const host = document.createElement('div');
		document.body.appendChild(host);
		dismiss(host, { onDismiss: () => undefined });
		const { editor } = mountEditor('a cat sits', {}, host);
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		expect(modal()).not.toBeNull();
	});

	it('opens where the host can create variables even if none exist yet', async () => {
		const onVariableDefChange = vi.fn();
		const { editor } = mountEditor('a cat sits', { variables: {}, onVariableDefChange });
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		expect(modal()).not.toBeNull();
	});
});

describe('typing << on Enter-created lines', () => {
	const lines = ['line1', 'line2', 'line3'];

	it('replaces the pair on the third line and leaves the caret after the chip', async () => {
		const { editor, values } = mountEditor(lines.join('\n'));
		await settle();
		const texts = toLines(editor, lines);
		setCaret(texts[2], 5);
		await typePair(editor);
		expect(values.at(-1)).toBe('line1\nline2\nline3<<');
		await saveModalWithText('storm');
		expect(values.at(-1)).toBe(`line1\nline2\nline3${STORM}`);
		expect(caretIsAfter(editor, editor.querySelector('.choice-group-container')!)).toBe(true);
		expect(caretOffset(editor)).toBe('line1\nline2\nline3'.length + STORM.length);
	});

	it('replaces the pair in the middle of the second line', async () => {
		const { editor, values } = mountEditor('line1\nline2 tail\nline3');
		await settle();
		const texts = toLines(editor, ['line1', 'line2 tail', 'line3']);
		setCaret(texts[1], 5);
		await typePair(editor);
		await saveModalWithText('storm');
		expect(values.at(-1)).toBe(`line1\nline2${STORM} tail\nline3`);
		expect(caretOffset(editor)).toBe('line1\nline2'.length + STORM.length);
	});

	it('keeps the literal pair and puts the caret after it on cancel', async () => {
		const { editor, values } = mountEditor(lines.join('\n'));
		await settle();
		const texts = toLines(editor, lines);
		setCaret(texts[2], 5);
		await typePair(editor);
		await cancelModal();
		expect(values.at(-1)).toBe('line1\nline2\nline3<<');
		expect(caretOffset(editor)).toBe('line1\nline2\nline3<<'.length);
		expect(document.activeElement).toBe(editor);
	});

	it('lands the caret after a condition inserted from the menu on the third line', async () => {
		const { editor, values, handle } = mountEditor(`${lines.join('\n')} end`);
		await settle();
		const texts = toLines(editor, [...lines.slice(0, 2), 'line3 end']);
		setCaret(texts[2], 5);
		const outside = document.createElement('button');
		document.body.appendChild(outside);
		setCaret(outside, 0);
		handle.insertCondition();
		await settle();
		expect(values.at(-1)).toBe(`line1\nline2\nline3${DEFAULT} end`);
		expect(caretIsAfter(editor, editor.querySelector('.choice-group-container')!)).toBe(true);
		expect(caretOffset(editor)).toBe('line1\nline2\nline3'.length + DEFAULT.length);
	});
});

describe('a typed << that moved before the condition was saved', () => {
	it('keeps a character typed after the pair and puts the chip where the pair was', async () => {
		const { editor, values } = mountEditor('a cat sits');
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		typeChar(editor, 'x');
		await settle();
		expect(values.at(-1)).toBe('a cat<<x sits');
		await saveModalWithText('storm');
		expect(values.at(-1)).toBe(`a cat${STORM}x sits`);
		expect(caretIsAfter(editor, editor.querySelector('.choice-group-container')!)).toBe(true);
	});

	it('appends the condition at the end and deletes nothing when the text changed under the pair', async () => {
		const { editor, values, handle } = mountEditor('a cat sits');
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		handle.$set({ value: 'zzz a cat sits' });
		await settle();
		await saveModalWithText('storm');
		expect(values.at(-1)).toBe(`zzz a cat sits${STORM}`);
		expect(caretIsAfter(editor, editor.querySelector('.choice-group-container')!)).toBe(true);
	});

	it('appends the condition at the end and keeps a shifted literal pair', async () => {
		const { editor, values, handle } = mountEditor('a cat sits');
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		handle.$set({ value: 'zz a cat<< sits' });
		await settle();
		await saveModalWithText('storm');
		expect(values.at(-1)).toBe(`zz a cat<< sits${STORM}`);
	});

	it('leaves the text and the caret alone on cancel when the pair is gone', async () => {
		const { editor, values, handle } = mountEditor('a cat sits');
		await settle();
		startAfter(editor, 5);
		await typePair(editor);
		handle.$set({ value: 'zzz a cat sits' });
		await settle();
		setCaret(editor.firstChild!, 4);
		const before = values.length;
		await cancelModal();
		expect(modal()).toBeNull();
		expect(values).toHaveLength(before);
		expect(editor.textContent).toBe('zzz a cat sits');
		expect(caretOffset(editor)).toBe(4);
	});
});
