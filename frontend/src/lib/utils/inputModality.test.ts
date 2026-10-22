// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { installInputModality } from './inputModality';

let stop: () => void;
let first: HTMLButtonElement;
let second: HTMLButtonElement;

const modality = () => document.documentElement.dataset.input;

function press(key: string, init: KeyboardEventInit = {}) {
	document.body.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, ...init }));
}

function release(key: string) {
	document.body.dispatchEvent(new KeyboardEvent('keyup', { key, bubbles: true }));
}

function pointerDown() {
	document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }));
}

beforeEach(() => {
	first = document.createElement('button');
	second = document.createElement('button');
	document.body.append(first, second);
	stop = installInputModality(document);
});

afterEach(() => {
	stop();
	document.body.innerHTML = '';
});

describe('input modality', () => {
	it('stays unset until the first input so a failed tracker never hides rings', () => {
		expect(modality()).toBeUndefined();
	});

	it('switches to pointer on pointerdown and back to keyboard on Tab', () => {
		pointerDown();
		expect(modality()).toBe('pointer');
		press('Tab');
		expect(modality()).toBe('keyboard');
	});

	it('treats Shift+Tab as keyboard navigation', () => {
		pointerDown();
		press('Tab', { shiftKey: true });
		expect(modality()).toBe('keyboard');
	});

	it.each<[string, KeyboardEventInit]>([
		['Enter', {}],
		[' ', {}],
		['Escape', {}],
		['g', {}],
		['w', {}],
		['Shift', { shiftKey: true }],
		['Control', { ctrlKey: true }],
		['Enter', { ctrlKey: true }],
		['k', { metaKey: true }],
		['Tab', { ctrlKey: true }],
		['ArrowDown', { altKey: true }]
	])('keeps pointer mode for app shortcut key %j %j', (key, init) => {
		pointerDown();
		press(key, init);
		first.focus();
		expect(modality()).toBe('pointer');
	});

	it('switches to keyboard when an arrow key moves focus', () => {
		pointerDown();
		first.focus();
		press('ArrowDown');
		second.focus();
		expect(modality()).toBe('keyboard');
	});

	it('flips when the focused element handles the arrow key itself and moves focus synchronously', () => {
		pointerDown();
		first.focus();
		first.addEventListener('keydown', () => second.focus());
		first.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true }));
		expect(document.activeElement).toBe(second);
		expect(modality()).toBe('keyboard');
	});

	it.each(['ArrowUp', 'ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown'])('counts %s as navigation when it moves focus', (key) => {
		pointerDown();
		press(key);
		second.focus();
		expect(modality()).toBe('keyboard');
	});

	it('stays in pointer mode when an arrow key does not move focus', () => {
		pointerDown();
		first.focus();
		press('ArrowDown');
		release('ArrowDown');
		expect(modality()).toBe('pointer');
	});

	it('ignores focus that moves after the arrow key was released', () => {
		pointerDown();
		press('ArrowDown');
		release('ArrowDown');
		second.focus();
		expect(modality()).toBe('pointer');
	});

	it('ignores focus that moves without any key', () => {
		pointerDown();
		second.focus();
		expect(modality()).toBe('pointer');
	});

	it('forgets a held arrow key when the pointer is pressed', () => {
		press('ArrowDown');
		pointerDown();
		second.focus();
		expect(modality()).toBe('pointer');
	});

	it('switches to keyboard when focus enters the page from outside after pointer use', () => {
		pointerDown();
		window.dispatchEvent(new Event('blur'));
		first.focus();
		expect(modality()).toBe('keyboard');
	});

	it('switches to keyboard when the window returns focus to a different element than it left', () => {
		pointerDown();
		first.focus();
		window.dispatchEvent(new Event('blur'));
		second.focus();
		expect(modality()).toBe('keyboard');
	});

	it('stays in pointer mode when the window returns focus to the element it left', () => {
		pointerDown();
		first.focus();
		window.dispatchEvent(new Event('blur'));
		first.blur();
		first.focus();
		expect(modality()).toBe('pointer');
	});

	it('stays in pointer mode when the pointer brings focus back into the page', () => {
		pointerDown();
		first.focus();
		window.dispatchEvent(new Event('blur'));
		pointerDown();
		second.focus();
		expect(modality()).toBe('pointer');
	});

	it('treats only the first focus after re-entering as coming from outside', () => {
		window.dispatchEvent(new Event('blur'));
		first.focus();
		pointerDown();
		second.focus();
		expect(modality()).toBe('pointer');
	});

	it('stops listening and clears the mode on teardown', () => {
		pointerDown();
		stop();
		expect(modality()).toBeUndefined();
		pointerDown();
		press('Tab');
		expect(modality()).toBeUndefined();
		stop = installInputModality(document);
	});
});
