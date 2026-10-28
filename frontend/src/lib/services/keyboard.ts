import { get } from 'svelte/store';
import { page } from '$app/stores';
import { keybindingsStore } from '$lib/stores/keybindings';
import { isTypingInField } from '$lib/utils/editableTarget';

let initialized = false;
let suppressGlobal = false;

export function init() {
	if (initialized) return;
	document.addEventListener('keydown', handleKeydown);
	initialized = true;
}

export function destroy() {
	if (!initialized) return;
	document.removeEventListener('keydown', handleKeydown);
	initialized = false;
}

export function suppressKeyboard() {
	suppressGlobal = true;
}

export function resumeKeyboard() {
	suppressGlobal = false;
}

export function shouldIgnoreEvent(e: KeyboardEvent): boolean {
	return isTypingInField(e) && !(e.ctrlKey || e.metaKey || e.altKey);
}

function getContext(pathname: string): string {
	if (pathname.startsWith('/generate')) return 'generate';
	if (pathname.startsWith('/history')) return 'history';
	if (pathname.startsWith('/models')) return 'models';
	if (pathname.startsWith('/admin')) return 'admin';
	return 'global';
}

function normalizeKey(e: KeyboardEvent): { key: string; modifiers: string } {
	const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;

	// For non-letter single characters (like ?, !, @, #), shift is implicit
	// in producing the character - don't treat it as a modifier
	const isImplicitShift = e.key.length === 1 && e.shiftKey && !/^[a-zA-Z]$/.test(e.key);

	const modifiers = [
		e.ctrlKey && 'ctrl',
		e.shiftKey && !isImplicitShift && 'shift',
		e.altKey && 'alt',
		e.metaKey && 'meta'
	]
		.filter(Boolean)
		.join(',');

	return { key, modifiers };
}

function handleKeydown(e: KeyboardEvent) {
	if (suppressGlobal || e.defaultPrevented) return;
	if (shouldIgnoreEvent(e)) return;

	const { key, modifiers } = normalizeKey(e);
	const state = get(keybindingsStore);
	const currentPath = get(page).url.pathname;
	const currentContext = getContext(currentPath);

	const match = state.bindings.find((b) => {
		if (!b.enabled || !b.key || !b.handler) return false;
		if (b.key.toLowerCase() !== key) return false;
		if ((b.modifiers || '') !== modifiers) return false;
		return b.context === 'global' || b.context === currentContext;
	});

	if (match && match.handler) {
		e.preventDefault();
		e.stopPropagation();
		match.handler();
	}
}
