// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { mount, unmount, flushSync, createRawSnippet } from 'svelte';
import { CHIP_POPOVER_SHEET_BREAKPOINT, CHIP_POPOVER_VIEWPORT_GUTTER } from '$lib/utils/chipPopover';

const { default: AnchoredPanel } = await import('$lib/components/AnchoredPanel.svelte');

const body = createRawSnippet(() => ({ render: () => `<button type="button" data-testid="inside">inside</button>` }));

let component: ReturnType<typeof mount> | undefined;
let anchor: HTMLButtonElement;
const originalWidth = window.innerWidth;

function setViewport(width: number) {
	Object.defineProperty(window, 'innerWidth', { value: width, configurable: true });
}

function mountPanel(onDismiss = vi.fn()) {
	anchor = document.createElement('button');
	document.body.appendChild(anchor);
	component = mount(AnchoredPanel, {
		target: document.body,
		props: { anchor, label: 'Test panel', onDismiss, children: body }
	});
	flushSync();
	return { onDismiss, panel: document.querySelector<HTMLElement>('[role="dialog"]')! };
}

function pointerDown(el: Element) {
	el.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true }));
	flushSync();
}

afterEach(() => {
	if (component) unmount(component);
	component = undefined;
	document.body.innerHTML = '';
	setViewport(originalWidth);
});

describe('AnchoredPanel', () => {
	it('is a bottom sheet at or below the sheet breakpoint', () => {
		setViewport(CHIP_POPOVER_SHEET_BREAKPOINT);
		const { panel } = mountPanel();
		expect(panel.style.left).toBe(`${CHIP_POPOVER_VIEWPORT_GUTTER}px`);
		expect(panel.style.right).toBe(`${CHIP_POPOVER_VIEWPORT_GUTTER}px`);
		expect(panel.style.bottom).toBe(`${CHIP_POPOVER_VIEWPORT_GUTTER}px`);
		expect(panel.style.width).toBe('');
	});

	it('is positioned against its anchor above the sheet breakpoint', () => {
		setViewport(CHIP_POPOVER_SHEET_BREAKPOINT + 1);
		const { panel } = mountPanel();
		expect(panel.style.width).not.toBe('');
		expect(panel.style.right).toBe('');
	});

	it('dismisses on a click outside the panel', () => {
		const { onDismiss } = mountPanel();
		pointerDown(document.body);
		expect(onDismiss).toHaveBeenCalledTimes(1);
		expect(onDismiss.mock.calls[0][0]).toBe('outside');
	});

	it('does not dismiss on a click inside the panel or on its trigger', () => {
		const { onDismiss, panel } = mountPanel();
		pointerDown(anchor);
		pointerDown(panel.querySelector('[data-testid="inside"]')!);
		expect(onDismiss).not.toHaveBeenCalled();
	});
});
