const NON_TEXT_INPUT_TYPES = new Set(['button', 'checkbox', 'color', 'file', 'hidden', 'image', 'radio', 'range', 'reset', 'submit']);
const TEXT_ROLE_SELECTOR = '[role="textbox"], [role="searchbox"], [role="combobox"]';
const NAVIGATION_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End']);
const NAVIGABLE_SELECTOR = 'input[type="range"], input[type="radio"], [role="slider"], [role="radio"], [role="spinbutton"]';

export function isEditableTarget(target: EventTarget | null): boolean {
	if (!(target instanceof Element)) return false;
	if (target instanceof HTMLInputElement) return !NON_TEXT_INPUT_TYPES.has(target.type);
	if (target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return true;
	const host = target.closest('[contenteditable]');
	if (host && host.getAttribute('contenteditable') !== 'false') return true;
	return target.closest(TEXT_ROLE_SELECTOR) !== null;
}

export function isTypingInField(event: KeyboardEvent): boolean {
	const origin = event.composedPath()[0] ?? event.target;
	if (event.isComposing || isEditableTarget(origin)) return true;
	return NAVIGATION_KEYS.has(event.key) && origin instanceof Element && origin.closest(NAVIGABLE_SELECTOR) !== null;
}
