import { describe, it, expect, afterEach, beforeEach } from 'vitest';
import { mount, unmount, flushSync, type Component } from 'svelte';

const { default: IconButton } = await import('../../src/lib/components/ui/IconButton.svelte');
const { default: TooltipWrappedControlsFixture } = await import('./fixtures/TooltipWrappedControlsFixture.svelte');
const { installInputModality } = await import('../../src/lib/utils/inputModality');

let target: HTMLDivElement;
let component: ReturnType<typeof mount> | null = null;
let stopInputModality: (() => void) | null = null;

function render<Props extends Record<string, any>>(view: Component<Props>, props: Props) {
	target = document.createElement('div');
	document.body.appendChild(target);
	component = mount(view, { target, props });
	flushSync();
}

function button(selector = 'button'): HTMLElement {
	return target.querySelector(selector) as HTMLElement;
}

function tooltips(): HTMLElement[] {
	return Array.from(document.body.querySelectorAll<HTMLElement>('[role="tooltip"]'));
}

async function wait(ms: number) {
	await new Promise((resolve) => setTimeout(resolve, ms));
	flushSync();
}

function hover(element: HTMLElement) {
	element.dispatchEvent(new MouseEvent('mouseenter'));
}

function unhover(element: HTMLElement) {
	element.dispatchEvent(new MouseEvent('mouseleave'));
	flushSync();
}

async function keyboardFocus(element: HTMLElement) {
	document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
	element.focus();
	await wait(0);
}

function blur(element: HTMLElement) {
	element.blur();
	flushSync();
}

beforeEach(() => {
	stopInputModality = installInputModality();
});

afterEach(() => {
	stopInputModality?.();
	stopInputModality = null;
	if (component) {
		unmount(component);
		component = null;
	}
	target?.remove();
	document.body.querySelectorAll('[role="tooltip"]').forEach((node) => node.remove());
});

describe('IconButton without tooltip props', () => {
	it('is a lone button that shows its label in the host tooltip instead of a native title', async () => {
		render(IconButton, { icon: 'trash', label: 'Delete' });

		expect(button().parentElement).toBe(target);
		expect(button().hasAttribute('title')).toBe(false);
		expect(button().getAttribute('aria-label')).toBe('Delete');
		expect(button().hasAttribute('aria-describedby')).toBe(false);

		hover(button());
		await wait(250);

		const [tooltip] = tooltips();
		expect(tooltip.textContent?.trim()).toBe('Delete');
		expect(button().getAttribute('aria-describedby')).toBe(tooltip.id);
	});

	it('drops its title inside a Tooltip and is described by it while it shows', async () => {
		render(TooltipWrappedControlsFixture, {});
		const iconButton = button('button[aria-label="Delete"]');

		expect(iconButton.hasAttribute('title')).toBe(false);
		expect(iconButton.hasAttribute('aria-describedby')).toBe(false);

		await keyboardFocus(iconButton);

		const [tooltip] = tooltips();
		expect(tooltip.textContent?.trim()).toBe('Delete the shot');
		expect(iconButton.getAttribute('aria-describedby')).toBe(tooltip.id);

		blur(iconButton);
		expect(tooltips()).toEqual([]);
		expect(iconButton.hasAttribute('aria-describedby')).toBe(false);
	});
});

describe('IconButton with tooltip or kbd', () => {
	it('renders the host tooltip itself instead of a native title', async () => {
		render(IconButton, { icon: 'film', label: 'Render', tooltip: 'Render the selected take' });
		const trigger = button();

		expect(trigger.hasAttribute('title')).toBe(false);
		expect(trigger.getAttribute('aria-label')).toBe('Render');
		expect(tooltips()).toEqual([]);

		hover(trigger);
		await wait(250);

		const [tooltip] = tooltips();
		expect(tooltip.textContent?.trim()).toBe('Render the selected take');
		expect(tooltip.hasAttribute('aria-hidden')).toBe(false);
		expect(trigger.getAttribute('aria-describedby')).toBe(tooltip.id);

		unhover(trigger);
		expect(tooltips()).toEqual([]);
		expect(trigger.hasAttribute('aria-describedby')).toBe(false);
	});

	it('falls back to the label and shows the shortcut, also to screen readers', async () => {
		render(IconButton, { icon: 'play', label: 'Play', kbd: ['Shift', 'Space'] });
		const trigger = button();

		expect(trigger.hasAttribute('title')).toBe(false);
		await keyboardFocus(trigger);

		const [tooltip] = tooltips();
		expect(tooltip.querySelector('span')?.textContent).toBe('Play');
		expect(Array.from(tooltip.querySelectorAll('kbd')).map((key) => key.textContent?.trim())).toEqual([
			'Shift',
			'Space'
		]);
		expect(tooltip.querySelector('.sr-only')?.textContent).toBe('Shift+Space');
		expect(trigger.getAttribute('aria-describedby')).toBe(tooltip.id);
	});

	it('shows at once on keyboard focus and hides on blur', async () => {
		render(IconButton, { icon: 'trash', label: 'Delete', tooltip: 'Delete the shot', kbd: 'Del' });
		const trigger = button();

		await keyboardFocus(trigger);
		expect(tooltips()).toHaveLength(1);

		blur(trigger);
		expect(tooltips()).toEqual([]);
	});

	it('does not show when the focus comes from a mouse press or a script after one', async () => {
		render(IconButton, { icon: 'trash', label: 'Delete', tooltip: 'Delete the shot' });
		const trigger = button();

		trigger.dispatchEvent(new Event('pointerdown', { bubbles: true }));
		trigger.focus();
		await wait(0);
		expect(tooltips()).toEqual([]);
		expect(trigger.hasAttribute('aria-describedby')).toBe(false);

		blur(trigger);
		document.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true }));
		trigger.focus();
		await wait(0);
		expect(tooltips()).toEqual([]);

		blur(trigger);
		await keyboardFocus(trigger);
		expect(tooltips()).toHaveLength(1);
	});

	it('passes its button props through without leaking tooltip or kbd into the DOM', () => {
		render(IconButton, {
			icon: 'trash',
			label: 'Delete',
			tooltip: 'Delete',
			kbd: ['Shift', 'Del'],
			disabled: true,
			class: 'ml-2',
			'data-testid': 'delete-shot'
		});

		expect(button().hasAttribute('disabled')).toBe(true);
		expect(button().className).toContain('ml-2');
		expect(button().getAttribute('data-testid')).toBe('delete-shot');
		expect(button().hasAttribute('tooltip')).toBe(false);
		expect(button().hasAttribute('kbd')).toBe(false);
	});
});

describe('Tooltip on keyboard focus', () => {
	it('describes a Button inside it', async () => {
		render(TooltipWrappedControlsFixture, {});
		const renderButton = Array.from(target.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Render')!;

		await keyboardFocus(renderButton);

		const [tooltip] = tooltips();
		expect(tooltip.textContent?.trim()).toBe('Render every take');
		expect(renderButton.getAttribute('aria-describedby')).toBe(tooltip.id);
	});

	it('shows for any focusable child and describes it', async () => {
		render(TooltipWrappedControlsFixture, {});
		const plain = button('.plain-child');

		await keyboardFocus(plain);

		const [tooltip] = tooltips();
		expect(tooltips().map((tooltip) => tooltip.textContent?.trim())).toEqual(['Plain child']);
		expect(plain.getAttribute('aria-describedby')).toBe(tooltip.id);
	});

	it('does not show when the focus follows a pointer press', async () => {
		render(TooltipWrappedControlsFixture, {});
		const plain = button('.plain-child');

		plain.dispatchEvent(new Event('pointerdown', { bubbles: true }));
		plain.focus();
		await wait(0);
		expect(tooltips()).toEqual([]);
		expect(plain.hasAttribute('aria-describedby')).toBe(false);
	});

	it('gives each tooltip its own id', async () => {
		render(TooltipWrappedControlsFixture, {});
		const iconButton = button('button[aria-label="Delete"]');
		const renderButton = Array.from(target.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Render')!;

		await keyboardFocus(iconButton);
		const first = iconButton.getAttribute('aria-describedby');
		await keyboardFocus(renderButton);
		const second = renderButton.getAttribute('aria-describedby');

		expect(first).toBeTruthy();
		expect(second).toBeTruthy();
		expect(first).not.toBe(second);
		expect(iconButton.hasAttribute('aria-describedby')).toBe(false);
	});
});
