// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createClassComponent } from 'svelte/legacy';
import Menu from '$lib/components/ui/Menu.svelte';
import { resetLayerStackForTests } from '$lib/actions/layerStack';

const items = [
	{ id: 'rename', label: 'Rename', icon: 'edit' },
	{ id: 'skip', label: 'Skip', disabled: true },
	{ id: 'duplicate', label: 'Duplicate', icon: 'copy' },
	{ id: 'delete', label: 'Delete', icon: 'trash', danger: true }
];

let target: HTMLElement;
let component: { $destroy: () => void } | undefined;

async function settle() {
	for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
}

async function mountMenu(onselect = vi.fn()) {
	component = createClassComponent({
		component: Menu as never,
		target,
		props: { label: 'More actions for X', menuLabel: 'Actions for X', items, onselect }
	});
	return onselect;
}

async function openMenu() {
	target.querySelector<HTMLButtonElement>('button[aria-label="More actions for X"]')!.click();
	await settle();
}

const menu = () => document.querySelector<HTMLElement>('[role="menu"]');
const menuItems = () => Array.from(document.querySelectorAll<HTMLButtonElement>('[role="menuitem"]'));

function key(el: Element, k: string) {
	el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }));
}

beforeEach(() => {
	resetLayerStackForTests();
	target = document.createElement('div');
	document.body.appendChild(target);
});

afterEach(() => {
	component?.$destroy();
	component = undefined;
	document.body.innerHTML = '';
});

describe('ui/Menu', () => {
	it('opens a labelled menu and focuses the first enabled item', async () => {
		await mountMenu();
		expect(menu()).toBeNull();
		await openMenu();

		expect(menu()?.getAttribute('aria-label')).toBe('Actions for X');
		expect(document.activeElement).toBe(menuItems()[0]);
	});

	it('moves focus with Arrow keys, wrapping and skipping disabled items', async () => {
		await mountMenu();
		await openMenu();
		const [rename, duplicate, del] = menuItems().filter((b) => !b.disabled);

		key(rename, 'ArrowDown');
		expect(document.activeElement).toBe(duplicate);
		key(duplicate, 'ArrowDown');
		expect(document.activeElement).toBe(del);
		key(del, 'ArrowDown');
		expect(document.activeElement).toBe(rename);
		key(rename, 'ArrowUp');
		expect(document.activeElement).toBe(del);
		key(del, 'Home');
		expect(document.activeElement).toBe(rename);
		key(rename, 'End');
		expect(document.activeElement).toBe(del);
	});

	it('closes on Escape and returns focus to the trigger', async () => {
		await mountMenu();
		await openMenu();

		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
		await settle();

		expect(menu()).toBeNull();
		expect(document.activeElement).toBe(target.querySelector('button[aria-label="More actions for X"]'));
	});

	it('closes on an outside pointerdown but not on the trigger', async () => {
		await mountMenu();
		await openMenu();

		target.querySelector('button')!.dispatchEvent(new Event('pointerdown', { bubbles: true }));
		expect(menu()).not.toBeNull();
		document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }));
		await settle();
		expect(menu()).toBeNull();
	});

	it('reports the chosen id and closes', async () => {
		const onselect = await mountMenu();
		await openMenu();

		menuItems().find((b) => b.textContent?.includes('Delete'))!.click();
		await settle();

		expect(onselect).toHaveBeenCalledWith('delete');
		expect(menu()).toBeNull();
	});
});
