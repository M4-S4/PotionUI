// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import menuNav, { focusFirstItem, menuItems } from './menuNav';

let menu: HTMLDivElement;
let stop: () => void;

function item(label: string, attributes: Record<string, string> = {}): HTMLButtonElement {
	const button = document.createElement('button');
	button.type = 'button';
	button.setAttribute('role', 'menuitem');
	button.textContent = label;
	for (const [name, value] of Object.entries(attributes)) button.setAttribute(name, value);
	return button;
}

function press(key: string) {
	(document.activeElement ?? menu).dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
}

const focusedLabel = () => document.activeElement?.textContent;

beforeEach(() => {
	menu = document.createElement('div');
	menu.setAttribute('role', 'menu');
	document.body.appendChild(menu);
	stop = menuNav(menu).destroy;
});

afterEach(() => {
	stop();
	document.body.innerHTML = '';
});

describe('menu items', () => {
	it('skips items that are not rendered, directly or through a hidden ancestor', () => {
		const group = document.createElement('div');
		group.style.display = 'none';
		const nested = item('Nested');
		group.appendChild(nested);
		menu.append(item('Own', { style: 'display: none' }), item('Attr', { hidden: '' }), group, item('Shown'));

		expect(menuItems(menu).map((el) => el.textContent)).toEqual(['Shown']);
	});

	it('skips disabled items as before', () => {
		menu.append(item('Off', { disabled: '' }), item('On'));

		expect(menuItems(menu).map((el) => el.textContent)).toEqual(['On']);
	});

	it('focuses the first rendered item', () => {
		menu.append(item('Own', { style: 'display: none' }), item('Shown'), item('Later'));

		focusFirstItem(menu);

		expect(focusedLabel()).toBe('Shown');
	});
});

describe('menuNav keyboard movement', () => {
	beforeEach(() => {
		menu.append(item('Hidden first', { style: 'display: none' }), item('One'), item('Hidden middle', { hidden: '' }), item('Two'), item('Three'));
	});

	it('moves between rendered rows only, wrapping at both ends', () => {
		focusFirstItem(menu);
		expect(focusedLabel()).toBe('One');

		press('ArrowDown');
		expect(focusedLabel()).toBe('Two');
		press('ArrowDown');
		expect(focusedLabel()).toBe('Three');
		press('ArrowDown');
		expect(focusedLabel()).toBe('One');
		press('ArrowUp');
		expect(focusedLabel()).toBe('Three');
	});

	it('lands on the first rendered row when focus is outside the menu', () => {
		const outside = document.createElement('button');
		document.body.appendChild(outside);
		outside.focus();

		menu.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true }));

		expect(focusedLabel()).toBe('One');
	});

	it('maps Home and End to the first and last rendered rows', () => {
		focusFirstItem(menu);
		press('End');
		expect(focusedLabel()).toBe('Three');
		press('Home');
		expect(focusedLabel()).toBe('One');
	});

	it('does nothing when no row is rendered', () => {
		menu.innerHTML = '';
		menu.append(item('Hidden', { style: 'display: none' }));

		const event = new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true });
		menu.dispatchEvent(event);

		expect(event.defaultPrevented).toBe(false);
		expect(document.activeElement).toBe(document.body);
	});
});
