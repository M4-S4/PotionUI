// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';

vi.mock('$lib/services/api', () => ({
	api: { getWorkspaces: vi.fn().mockResolvedValue({ success: true, data: [] }) }
}));

const { tabsStore } = await import('$lib/stores/tabs');
const { default: TabBar } = await import('../../src/routes/generate/components/TabBar.svelte');
const { createClassComponent } = await import('svelte/legacy');

let destroy: (() => void) | undefined;

function mountBar() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({ component: TabBar as never, target });
	destroy = () => {
		component.$destroy();
		target.remove();
	};
}

async function settle() {
	for (let i = 0; i < 6; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

const trigger = () => document.body.querySelector<HTMLButtonElement>('.workspace-menu-container button')!;
const menu = () => document.body.querySelector('.workspace-menu-container .absolute');

function pointerdown(el: Element) {
	el.dispatchEvent(new Event('pointerdown', { bubbles: true, cancelable: true }));
}

async function pressTrigger() {
	pointerdown(trigger());
	trigger().click();
	await settle();
}

beforeEach(() => {
	tabsStore.reset();
	tabsStore.addTab();
	mountBar();
});

afterEach(() => {
	destroy?.();
	destroy = undefined;
	tabsStore.reset();
	document.body.innerHTML = '';
});

describe('TabBar workspace menu dismissal', () => {
	it('opens on the trigger and ends closed when the trigger is pressed again', async () => {
		await pressTrigger();
		expect(menu()).not.toBeNull();
		await pressTrigger();
		expect(menu()).toBeNull();
	});

	it('closes on an outside press', async () => {
		await pressTrigger();
		pointerdown(document.body);
		await settle();
		expect(menu()).toBeNull();
	});

	it('stays open on a press inside the menu', async () => {
		await pressTrigger();
		pointerdown(menu()!);
		await settle();
		expect(menu()).not.toBeNull();
	});

	it('closes on Escape', async () => {
		await pressTrigger();
		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
		await settle();
		expect(menu()).toBeNull();
	});
});
