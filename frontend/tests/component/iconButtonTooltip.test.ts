import { describe, it, expect, afterEach } from 'vitest';
import { mount, unmount, flushSync, type Component } from 'svelte';

const { default: IconButton } = await import('../../src/lib/components/ui/IconButton.svelte');
const { default: TooltipWrappedControlsFixture } = await import('./fixtures/TooltipWrappedControlsFixture.svelte');

let target: HTMLDivElement;
let component: ReturnType<typeof mount> | null = null;

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

function wrapperOf(element: HTMLElement): HTMLElement {
	return element.parentElement as HTMLElement;
}

async function wait(ms: number) {
	await new Promise((resolve) => setTimeout(resolve, ms));
	flushSync();
}

function hover(element: HTMLElement) {
	wrapperOf(element).dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
}

function unhover(element: HTMLElement) {
	wrapperOf(element).dispatchEvent(new MouseEvent('mouseleave', { bubbles: true }));
	flushSync();
}

function keyboardFocus(element: HTMLElement) {
	document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
	element.focus();
	flushSync();
}

function blur(element: HTMLElement) {
	element.blur();
	flushSync();
}

afterEach(() => {
	if (component) {
		unmount(component);
		component = null;
	}
	target?.remove();
	document.body.querySelectorAll('[role="tooltip"]').forEach((node) => node.remove());
});

describe('IconButton without tooltip props', () => {
	it('renders exactly as before: a lone button titled with its label', () => {
		render(IconButton, { icon: 'trash', label: 'Delete' });

		expect(button().parentElement).toBe(target);
		expect(button().getAttribute('title')).toBe('Delete');
		expect(button().getAttribute('aria-label')).toBe('Delete');
		expect(button().hasAttribute('aria-describedby')).toBe(false);
	});

	it('drops its title inside a Tooltip and is described by it while it shows', () => {
		render(TooltipWrappedControlsFixture, {});
		const iconButton = button('button[aria-label="Delete"]');

		expect(iconButton.hasAttribute('title')).toBe(false);
		expect(iconButton.hasAttribute('aria-describedby')).toBe(false);

		keyboardFocus(iconButton);

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

	it('falls back to the label and shows the shortcut, also to screen readers', () => {
		render(IconButton, { icon: 'play', label: 'Play', kbd: ['Shift', 'Space'] });
		const trigger = button();

		expect(trigger.hasAttribute('title')).toBe(false);
		keyboardFocus(trigger);

		const [tooltip] = tooltips();
		expect(tooltip.querySelector('span')?.textContent).toBe('Play');
		expect(Array.from(tooltip.querySelectorAll('kbd')).map((key) => key.textContent?.trim())).toEqual([
			'Shift',
			'Space'
		]);
		expect(tooltip.querySelector('.sr-only')?.textContent).toBe('Shift+Space');
		expect(trigger.getAttribute('aria-describedby')).toBe(tooltip.id);
	});

	it('shows at once on keyboard focus and hides on blur', () => {
		render(IconButton, { icon: 'trash', label: 'Delete', tooltip: 'Delete the shot', kbd: 'Del' });
		const trigger = button();

		keyboardFocus(trigger);
		expect(tooltips()).toHaveLength(1);

		blur(trigger);
		expect(tooltips()).toEqual([]);
	});

	it('does not show when the focus comes from a mouse press or a script after one', () => {
		render(IconButton, { icon: 'trash', label: 'Delete', tooltip: 'Delete the shot' });
		const trigger = button();

		trigger.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
		trigger.focus();
		flushSync();
		expect(tooltips()).toEqual([]);
		expect(trigger.hasAttribute('aria-describedby')).toBe(false);

		blur(trigger);
		document.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true }));
		trigger.focus();
		flushSync();
		expect(tooltips()).toEqual([]);

		blur(trigger);
		keyboardFocus(trigger);
		expect(tooltips()).toHaveLength(1);
	});

	it('passes its button props through', () => {
		render(IconButton, { icon: 'trash', label: 'Delete', tooltip: 'Delete', disabled: true, class: 'ml-2' });

		expect(button().hasAttribute('disabled')).toBe(true);
		expect(button().className).toContain('ml-2');
	});
});

describe('Tooltip on keyboard focus', () => {
	it('describes a Button inside it', () => {
		render(TooltipWrappedControlsFixture, {});
		const renderButton = Array.from(target.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Render')!;

		keyboardFocus(renderButton);

		const [tooltip] = tooltips();
		expect(tooltip.textContent?.trim()).toBe('Render every take');
		expect(renderButton.getAttribute('aria-describedby')).toBe(tooltip.id);
	});

	it('shows for any focusable child, which it cannot describe', () => {
		render(TooltipWrappedControlsFixture, {});
		const plain = button('.plain-child');

		keyboardFocus(plain);

		expect(tooltips().map((tooltip) => tooltip.textContent?.trim())).toEqual(['Plain child']);
		expect(plain.hasAttribute('aria-describedby')).toBe(false);
	});

	it('gives each tooltip its own id', () => {
		render(TooltipWrappedControlsFixture, {});
		const iconButton = button('button[aria-label="Delete"]');
		const renderButton = Array.from(target.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Render')!;

		keyboardFocus(iconButton);
		const first = iconButton.getAttribute('aria-describedby');
		keyboardFocus(renderButton);
		const second = renderButton.getAttribute('aria-describedby');

		expect(first).toBeTruthy();
		expect(second).toBeTruthy();
		expect(first).not.toBe(second);
		expect(iconButton.hasAttribute('aria-describedby')).toBe(false);
	});
});
