// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';

const { default: delegatedTooltip } = await import('$lib/actions/delegatedTooltip');

let cleanup: Array<() => void> = [];

function setup() {
	const root = document.createElement('div');
	root.innerHTML = '<p>text <span data-tooltip="shuffles each time" tabindex="0"><b>mood</b></span></p><span id="plain">x</span>';
	document.body.appendChild(root);
	const action = delegatedTooltip(root);
	cleanup.push(() => {
		action.destroy();
		root.remove();
	});
	return {
		root,
		action,
		chip: root.querySelector<HTMLElement>('[data-tooltip]')!,
		inner: root.querySelector<HTMLElement>('b')!,
		plain: root.querySelector<HTMLElement>('#plain')!
	};
}

const bubbleText = () => document.body.querySelector('[role="tooltip"]')?.textContent?.trim() ?? null;

afterEach(() => {
	cleanup.forEach((fn) => fn());
	cleanup = [];
	vi.useRealTimers();
});

describe('delegatedTooltip', () => {
	it('shows the attribute value after the hover delay when a descendant is hovered', async () => {
		vi.useFakeTimers();
		const { inner } = setup();
		inner.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
		expect(bubbleText()).toBeNull();
		vi.advanceTimersByTime(250);
		expect(bubbleText()).toBe('shuffles each time');
	});

	it('hides when the pointer leaves the chip', () => {
		vi.useFakeTimers();
		const { chip, plain } = setup();
		chip.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
		vi.advanceTimersByTime(250);
		chip.dispatchEvent(new MouseEvent('mouseout', { bubbles: true, relatedTarget: plain }));
		expect(bubbleText()).toBeNull();
	});

	it('ignores elements without the attribute', () => {
		vi.useFakeTimers();
		const { plain } = setup();
		plain.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
		vi.advanceTimersByTime(250);
		expect(bubbleText()).toBeNull();
	});

	it('shows on focus, describes the chip, and hides on Escape', () => {
		const { chip } = setup();
		chip.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
		const bubble = document.body.querySelector('[role="tooltip"]')!;
		expect(bubble.textContent?.trim()).toBe('shuffles each time');
		expect(chip.getAttribute('aria-describedby')).toBe(bubble.id);
		chip.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
		expect(bubbleText()).toBeNull();
		expect(chip.hasAttribute('aria-describedby')).toBe(false);
	});

	it('hides when the anchor is replaced in the DOM', async () => {
		const { root, chip } = setup();
		chip.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
		expect(bubbleText()).toBe('shuffles each time');
		root.innerHTML = '<p>streamed</p>';
		await new Promise((r) => setTimeout(r, 0));
		expect(bubbleText()).toBeNull();
	});

	it('leaves no bubble after destroy while open', () => {
		const { chip, action } = setup();
		chip.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
		action.destroy();
		expect(bubbleText()).toBeNull();
	});

	it('hides on focusout', () => {
		const { chip } = setup();
		chip.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
		chip.dispatchEvent(new FocusEvent('focusout', { bubbles: true }));
		expect(bubbleText()).toBeNull();
	});

	it('keeps the bubble when the pointer moves to a child of the chip', () => {
		const { chip, inner } = setup();
		chip.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
		chip.dispatchEvent(new MouseEvent('mouseout', { bubbles: true, relatedTarget: inner }));
		expect(bubbleText()).toBe('shuffles each time');
	});

	it('restores a prior aria-describedby after hide', () => {
		const { chip } = setup();
		chip.setAttribute('aria-describedby', 'other');
		chip.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
		expect(chip.getAttribute('aria-describedby')).not.toBe('other');
		chip.dispatchEvent(new FocusEvent('focusout', { bubbles: true }));
		expect(chip.getAttribute('aria-describedby')).toBe('other');
	});
});
