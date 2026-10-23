// @vitest-environment jsdom
//
// InlinePopoverChip's popover used to render `absolute`, inside the chip —
// clipped whenever an ancestor (PromptSegment.svelte's `.card`) sets
// `overflow: hidden`. This proves the portal fix: the popover is a real
// `<body>`-level, `position: fixed` node while open, a click inside it
// (through the portal, not through `chipRef`) doesn't dismiss it, and it's
// cleaned up on both close and unmount.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mount, unmount, flushSync, createRawSnippet } from 'svelte';

import { CHIP_POPOVER_WIDTH } from '$lib/utils/chipPopover';

const { default: InlinePopoverChip } = await import('$lib/components/InlinePopoverChip.svelte');

let target: HTMLDivElement;
let component: ReturnType<typeof mount>;

const labelSnippet = createRawSnippet(() => ({
	render: () => `<span>trigger label</span>`
}));

const popoverSnippet = createRawSnippet(() => ({
	render: () => `<button type="button" data-testid="inside-popover">inside</button>`
}));

function mountChip(onremove = vi.fn()) {
	target = document.createElement('div');
	document.body.appendChild(target);
	component = mount(InlinePopoverChip, {
		target,
		props: {
			popoverLabel: 'Test popover',
			label: labelSnippet,
			popover: popoverSnippet,
			onremove
		}
	});
	flushSync();
}

function openChip() {
	const trigger = target.querySelectorAll('button')[0];
	trigger.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
	flushSync();
}

beforeEach(() => {
	if (!Element.prototype.animate) {
		Element.prototype.animate = () => {
			const animation = {
				cancel() {},
				finished: Promise.resolve(),
				onfinish: null as (() => void) | null,
				currentTime: 0,
				playbackRate: 1
			};
			setTimeout(() => animation.onfinish?.(), 0);
			return animation as never;
		};
	}
});

afterEach(() => {
	if (component) unmount(component);
	target?.remove();
	document.body.innerHTML = '';
	vi.clearAllMocks();
});

describe('InlinePopoverChip portal popover', () => {
	it('renders the open popover as a fixed, body-level node', () => {
		mountChip();
		openChip();

		const popover = document.querySelector('[role="dialog"]');
		expect(popover).not.toBeNull();
		expect(popover?.parentElement).toBe(document.body);
		expect(popover?.classList.contains('fixed')).toBe(true);
		// Not a descendant of the chip anymore — that's the whole point of the portal.
		expect(target.contains(popover)).toBe(false);
	});

	it('does not close on a pointerdown inside the portaled popover', () => {
		mountChip();
		openChip();

		const inside = document.querySelector('[data-testid="inside-popover"]');
		expect(inside).not.toBeNull();
		inside?.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
		flushSync();

		expect(document.querySelector('[role="dialog"]')).not.toBeNull();
	});

	it('closes on a pointerdown outside the chip and the popover', () => {
		mountChip();
		openChip();
		expect(document.querySelector('[role="dialog"]')).not.toBeNull();

		document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
		flushSync();

		expect(document.querySelector('[role="dialog"]')).toBeNull();
	});

	it('closes on Escape', () => {
		mountChip();
		openChip();
		expect(document.querySelector('[role="dialog"]')).not.toBeNull();

		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
		flushSync();

		expect(document.querySelector('[role="dialog"]')).toBeNull();
	});

	it('clamps a popover taller than the viewport room to that room and scrolls the rest', () => {
		// scrollHeight always reads 0 in jsdom's layout-less environment; stub it
		// so the popover measures as "tall content" the way a long option list
		// or long variable values would in a real browser.
		const scrollHeightSpy = vi.spyOn(Element.prototype, 'scrollHeight', 'get').mockReturnValue(900);
		Object.defineProperty(window, 'innerHeight', { value: 400, configurable: true });

		mountChip();
		const chip = target.querySelector('.inline-popover-chip') as HTMLElement;
		// Pinned low, with only 20px of room below and 350px above -- the
		// popover has to flip up, and 350px of room is far short of the 900px
		// of content, so it must clamp rather than run off the top.
		chip.getBoundingClientRect = () =>
			({ top: 350, bottom: 380, left: 200, right: 260, width: 60, height: 30, x: 200, y: 350, toJSON() {} }) as DOMRect;

		openChip();

		const popover = document.querySelector('[role="dialog"]') as HTMLElement;
		expect(popover).not.toBeNull();
		expect(popover.style.overflowY).toBe('auto');
		const maxHeight = parseFloat(popover.style.maxHeight);
		expect(maxHeight).toBeGreaterThan(0);
		// Room above the trigger is 350px -- the clamp must fit inside it.
		expect(maxHeight).toBeLessThanOrEqual(350);

		scrollHeightSpy.mockRestore();
	});

	it('removes the portaled popover from the DOM when the chip unmounts while open', () => {
		mountChip();
		openChip();
		expect(document.querySelector('[role="dialog"]')).not.toBeNull();

		unmount(component);
		component = undefined as unknown as ReturnType<typeof mount>;

		expect(document.querySelector('[role="dialog"]')).toBeNull();
	});

	it('uses the shared chip popover width on desktop', () => {
		Object.defineProperty(window, 'innerWidth', { value: 1440, configurable: true });
		mountChip();
		openChip();
		const popover = document.querySelector('[role="dialog"]') as HTMLElement;
		expect(popover.style.width).toBe(`${CHIP_POPOVER_WIDTH}px`);
	});

	it('becomes a full-width bottom sheet on a phone', () => {
		Object.defineProperty(window, 'innerWidth', { value: 390, configurable: true });
		mountChip();
		openChip();
		const popover = document.querySelector('[role="dialog"]') as HTMLElement;
		expect(popover.style.left).toBe('8px');
		expect(popover.style.right).toBe('8px');
		expect(popover.style.bottom).toBe('8px');
		expect(popover.style.width).toBe('auto');
		Object.defineProperty(window, 'innerWidth', { value: 1024, configurable: true });
	});

	it('ignores a pointerdown inside a nested chip layer', () => {
		mountChip();
		openChip();
		const layer = document.createElement('div');
		layer.setAttribute('data-chip-layer', '');
		document.body.appendChild(layer);
		layer.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
		flushSync();
		expect(document.querySelector('[role="dialog"]')).not.toBeNull();
	});

	it('keeps open when Escape was already handled by a nested control', () => {
		mountChip();
		openChip();
		const event = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
		event.preventDefault();
		window.dispatchEvent(event);
		flushSync();
		expect(document.querySelector('[role="dialog"]')).not.toBeNull();
	});

	function mountInEditor() {
		const host = document.createElement('div');
		host.setAttribute('contenteditable', 'true');
		host.setAttribute('tabindex', '-1');
		document.body.appendChild(host);
		target = document.createElement('div');
		host.appendChild(target);
		component = mount(InlinePopoverChip, {
			target,
			props: { popoverLabel: 'Test popover', label: labelSnippet, popover: popoverSnippet }
		});
		flushSync();
		return host;
	}

	it('closes only the innermost layer on Escape', () => {
		mountChip();
		openChip();
		const outer = vi.fn();
		document.addEventListener('keydown', outer);
		document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
		flushSync();
		document.removeEventListener('keydown', outer);
		expect(document.querySelector('[role="dialog"]')).toBeNull();
		expect(outer).not.toHaveBeenCalled();
	});

	it('returns focus and the caret to the editor after the chip when the popover closes', () => {
		const host = mountInEditor();
		openChip();
		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', cancelable: true }));
		flushSync();
		expect(document.activeElement).toBe(host);
		const range = window.getSelection()!.getRangeAt(0);
		expect(range.collapsed).toBe(true);
		const chip = target.querySelector('.inline-popover-chip')!;
		expect(range.startContainer === chip.parentNode || chip.compareDocumentPosition(range.startContainer) !== 0).toBe(true);
	});

	it('puts the caret after the non-editable container that holds the chip, not inside it', () => {
		const host = document.createElement('div');
		host.setAttribute('contenteditable', 'true');
		host.setAttribute('tabindex', '-1');
		host.append('before ');
		document.body.appendChild(host);
		target = document.createElement('div');
		target.setAttribute('contenteditable', 'false');
		host.appendChild(target);
		host.append(' after');
		component = mount(InlinePopoverChip, {
			target,
			props: { popoverLabel: 'Test popover', label: labelSnippet, popover: popoverSnippet }
		});
		flushSync();
		openChip();
		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', cancelable: true }));
		flushSync();
		const range = window.getSelection()!.getRangeAt(0);
		expect(range.collapsed).toBe(true);
		expect(range.startContainer).toBe(host);
		expect(range.startOffset).toBe(Array.from(host.childNodes).indexOf(target) + 1);
	});

	it('does not steal focus when closed by a pointerdown outside', () => {
		const host = mountInEditor();
		const other = document.createElement('input');
		document.body.appendChild(other);
		openChip();
		other.focus();
		other.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
		flushSync();
		expect(document.querySelector('[role="dialog"]')).toBeNull();
		expect(document.activeElement).toBe(other);
		expect(document.activeElement).not.toBe(host);
	});

	it('renders no popover when detached and still toggles open', () => {
		target = document.createElement('div');
		document.body.appendChild(target);
		component = mount(InlinePopoverChip, {
			target,
			props: { popoverLabel: 'Test popover', label: labelSnippet, detached: true }
		});
		flushSync();
		openChip();
		expect(document.querySelector('[role="dialog"]')).toBeNull();
	});
});
