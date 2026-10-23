import { settle } from './settle';

export function setCaret(node: Node, offset: number): void {
	const range = document.createRange();
	range.setStart(node, offset);
	range.collapse(true);
	const selection = window.getSelection()!;
	selection.removeAllRanges();
	selection.addRange(range);
	document.dispatchEvent(new Event('selectionchange'));
}

export function typeChar(
	editor: HTMLElement,
	character: string,
	init: InputEventInit = {},
	origin: HTMLElement = editor
): void {
	const range = window.getSelection()!.getRangeAt(0);
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

export async function typeConditionTrigger(editor: HTMLElement): Promise<void> {
	const text = editor.lastChild as Text;
	setCaret(text, text.data.length);
	typeChar(editor, '<');
	await settle();
	typeChar(editor, '<');
	await settle();
}

export const conditionModal = () =>
	document.querySelector<HTMLElement>('[role="dialog"][aria-labelledby="condition-modal-title"]');

export const requireConditionModal = () => conditionModal()!;
