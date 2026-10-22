// @vitest-environment jsdom
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { createClassComponent } from 'svelte/legacy';
import Menu from '$lib/components/ui/Menu.svelte';
import { resetLayerStackForTests } from '$lib/actions/layerStack';
import { installInputModality } from '$lib/utils/inputModality';

const items = [
	{ id: 'rename', label: 'Rename' },
	{ id: 'duplicate', label: 'Duplicate' },
	{ id: 'delete', label: 'Delete' }
];

let target: HTMLElement;
let component: { $destroy: () => void } | undefined;
let stop: () => void;

async function settle() {
	for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
}

const modality = () => document.documentElement.dataset.input;
const menuItems = () => Array.from(document.querySelectorAll<HTMLButtonElement>('[role="menuitem"]'));

beforeEach(() => {
	resetLayerStackForTests();
	target = document.createElement('div');
	document.body.appendChild(target);
	stop = installInputModality(document);
});

afterEach(() => {
	stop();
	component?.$destroy();
	component = undefined;
	document.body.innerHTML = '';
});

describe('input modality inside ui/Menu', () => {
	it('flips to keyboard when an arrow key moves focus between rows of a menu opened with the pointer', async () => {
		component = createClassComponent({ component: Menu as never, target, props: { label: 'More', items } });
		const trigger = target.querySelector<HTMLButtonElement>('button[aria-label="More"]')!;
		trigger.dispatchEvent(new Event('pointerdown', { bubbles: true }));
		trigger.click();
		await settle();

		const [first, second] = menuItems();
		expect(document.activeElement).toBe(first);
		expect(modality()).toBe('pointer');

		first.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true }));
		expect(document.activeElement).toBe(second);
		expect(modality()).toBe('keyboard');
	});
});
