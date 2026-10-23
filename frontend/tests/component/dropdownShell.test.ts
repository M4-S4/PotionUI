import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mount, unmount, flushSync, createRawSnippet } from 'svelte';

const { default: DropdownShell } = await import('$lib/components/drawer/DropdownShell.svelte');

function rect(left: number, top: number, width: number, height: number): DOMRect {
	return {
		left,
		top,
		width,
		height,
		right: left + width,
		bottom: top + height,
		x: left,
		y: top,
		toJSON: () => ({})
	} as DOMRect;
}

function element(box: DOMRect, attrs: Record<string, string> = {}): HTMLElement {
	const el = document.createElement('div');
	for (const [name, value] of Object.entries(attrs)) el.setAttribute(name, value);
	el.getBoundingClientRect = () => box;
	document.body.appendChild(el);
	return el;
}

const content = createRawSnippet(() => ({ render: () => '<p>panel content</p>' }));

let instance: ReturnType<typeof mount> | undefined;
let host: HTMLElement;

function open(props: Record<string, unknown>) {
	host = document.createElement('section');
	document.body.appendChild(host);
	const onClose = vi.fn();
	instance = mount(DropdownShell as never, {
		target: host,
		props: { label: 'Sessions', onClose, children: content, ...props }
	});
	flushSync();
	return onClose;
}

function panel(): HTMLElement {
	return document.body.querySelector<HTMLElement>('[role="dialog"][aria-label="Sessions"]')!;
}

function pointerDown(el: Element) {
	el.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true }));
	flushSync();
}

beforeEach(() => {
	Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1440 });
	Object.defineProperty(window, 'innerHeight', { configurable: true, value: 900 });
});

afterEach(() => {
	if (instance) unmount(instance);
	instance = undefined;
	document.body.innerHTML = '';
});

describe('DropdownShell placement', () => {
	it('renders at the page root, outside the scrolling form column', () => {
		const column = element(rect(56, 69, 408, 750), { style: 'overflow-y: auto' });
		const anchor = element(rect(68, 131, 384, 42));
		column.appendChild(anchor);
		open({ anchor });
		expect(panel().parentElement).toBe(document.body);
		expect(column.contains(panel())).toBe(false);
		expect(host.contains(panel())).toBe(false);
		expect(panel().textContent).toBe('panel content');
	});

	it('opens 440px wide, 6px under the anchor at its left edge, and stops above the generation panel', () => {
		element(rect(0, 818, 1440, 82), { 'data-menu-dock': '' });
		const anchor = element(rect(460, 73, 300, 40));
		open({ anchor });
		const style = panel().getAttribute('style') ?? '';
		expect(style).toContain('top: 119px');
		expect(style).toContain('left: 460px');
		expect(style).toContain('width: 440px');
		expect(style).toContain('max-height: 691px');
	});

	it('stays on screen next to the right edge and narrows to a small viewport', () => {
		const anchor = element(rect(1200, 73, 200, 40));
		open({ anchor });
		expect(panel().getAttribute('style')).toContain('left: 992px');
		unmount(instance!);
		instance = undefined;
		Object.defineProperty(window, 'innerWidth', { configurable: true, value: 400 });
		open({ anchor: element(rect(120, 73, 200, 40)) });
		const style = panel().getAttribute('style') ?? '';
		expect(style).toContain('width: 384px');
		expect(style).toContain('left: 8px');
	});

	it('fills the screen without a computed position when full screen', () => {
		const anchor = element(rect(16, 600, 358, 42));
		open({ anchor, fullscreen: true });
		expect(panel().classList.contains('is-fullscreen')).toBe(true);
		const style = panel().getAttribute('style') ?? '';
		expect(style).not.toContain('top:');
		expect(style).not.toContain('left:');
		expect(style).not.toContain('max-height');
	});
});

describe('DropdownShell dismissal', () => {
	it('closes on a pointer down outside', () => {
		const anchor = element(rect(68, 131, 384, 42));
		const outside = element(rect(600, 300, 100, 100));
		const onClose = open({ anchor });
		pointerDown(outside);
		expect(onClose).toHaveBeenCalledTimes(1);
	});

	it('ignores a pointer down on its anchor and inside a modal opened above it', () => {
		const anchor = element(rect(68, 131, 384, 42));
		const modal = element(rect(400, 200, 400, 300), { 'aria-modal': 'true', role: 'alertdialog' });
		const onClose = open({ anchor });
		pointerDown(anchor);
		pointerDown(modal);
		pointerDown(panel());
		expect(onClose).not.toHaveBeenCalled();
	});

	it('closes on Escape from inside the panel', () => {
		const anchor = element(rect(68, 131, 384, 42));
		const onClose = open({ anchor });
		panel().dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
		expect(onClose).toHaveBeenCalledTimes(1);
	});
});
