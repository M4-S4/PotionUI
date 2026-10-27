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

	it('focuses the first item only once the menu is positioned and visible', async () => {
		const styles: string[] = [];
		const focus = HTMLElement.prototype.focus;
		HTMLElement.prototype.focus = function (this: HTMLElement, ...args: Parameters<HTMLElement['focus']>) {
			if (this.getAttribute('role') === 'menuitem') styles.push(this.closest('[role="menu"]')?.getAttribute('style') ?? '');
			return focus.apply(this, args);
		};
		try {
			await mountMenu();
			await openMenu();
		} finally {
			HTMLElement.prototype.focus = focus;
		}

		expect(styles.length).toBeGreaterThan(0);
		for (const style of styles) expect(style).not.toContain('visibility: hidden');
		expect(document.activeElement).toBe(menuItems()[0]);
	});

	it('lands focus on the first enabled item when the trigger is activated from the keyboard', async () => {
		await mountMenu();
		const trigger = target.querySelector<HTMLButtonElement>('button[aria-label="More actions for X"]')!;
		trigger.focus();
		trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
		trigger.click();
		await settle();

		expect(document.activeElement).toBe(menuItems().find((b) => !b.disabled));
		expect(document.activeElement?.textContent).toContain('Rename');
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

	it('anchors to an explicit point instead of its trigger', async () => {
		const rect = { x: 120, y: 80, left: 120, right: 120, top: 80, bottom: 80, width: 0, height: 0, toJSON: () => ({}) };
		component = createClassComponent({
			component: Menu as never,
			target,
			props: {
				label: 'More actions for X',
				items,
				align: 'left',
				anchor: { getBoundingClientRect: () => rect },
				open: true
			}
		});
		await settle();

		const style = menu()!.getAttribute('style') ?? '';
		expect(style).toContain('left: 120px');
		expect(style).toContain('top: 84px');
	});
});
