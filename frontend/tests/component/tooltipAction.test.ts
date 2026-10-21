// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';

const { default: tooltip } = await import('$lib/actions/tooltip');
const { default: TooltipWrapFixture } = await import('./fixtures/TooltipWrapFixture.svelte');
const { createClassComponent } = await import('svelte/legacy');

let cleanup: Array<() => void> = [];

function attach(param: Parameters<typeof tooltip>[1], tag = 'button') {
	const node = document.createElement(tag);
	node.textContent = 'x';
	document.body.appendChild(node);
	const action = tooltip(node, param);
	cleanup.push(() => {
		action.destroy();
		node.remove();
	});
	return { node, action };
}

const bubbleText = () => document.body.querySelector('[role="tooltip"]')?.textContent?.trim() ?? null;

afterEach(() => {
	cleanup.forEach((fn) => fn());
	cleanup = [];
	vi.useRealTimers();
});

describe('tooltip action', () => {
	it('shows after the hover delay and hides on mouseleave', () => {
		vi.useFakeTimers();
		const { node } = attach('Copy');
		node.dispatchEvent(new MouseEvent('mouseenter'));
		expect(bubbleText()).toBeNull();
		vi.advanceTimersByTime(250);
		expect(bubbleText()).toBe('Copy');
		node.dispatchEvent(new MouseEvent('mouseleave'));
		expect(bubbleText()).toBeNull();
	});

	it('shows immediately on keyboard focus and describes the element while open', () => {
		const { node } = attach('Delete item');
		node.dispatchEvent(new FocusEvent('focusin'));
		const bubble = document.body.querySelector('[role="tooltip"]')!;
		expect(bubble.textContent?.trim()).toBe('Delete item');
		expect(node.getAttribute('aria-describedby')).toBe(bubble.id);
		node.dispatchEvent(new FocusEvent('focusout'));
		expect(bubbleText()).toBeNull();
		expect(node.hasAttribute('aria-describedby')).toBe(false);
	});

	it('closes on Escape', () => {
		const { node } = attach('Delete item');
		node.dispatchEvent(new FocusEvent('focusin'));
		node.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
		expect(bubbleText()).toBeNull();
	});

	it('shows nothing for empty text', () => {
		const { node } = attach(undefined);
		node.dispatchEvent(new FocusEvent('focusin'));
		expect(bubbleText()).toBeNull();
	});

	it('truncateOnly shows only when the content overflows', () => {
		const { node } = attach({ text: 'A very long name', truncateOnly: true }, 'span');
		node.dispatchEvent(new FocusEvent('focusin'));
		expect(bubbleText()).toBeNull();
		Object.defineProperty(node, 'scrollWidth', { value: 200, configurable: true });
		Object.defineProperty(node, 'clientWidth', { value: 100, configurable: true });
		node.dispatchEvent(new FocusEvent('focusin'));
		expect(bubbleText()).toBe('A very long name');
	});

	it('follows a changed text while open and removes the bubble on destroy', () => {
		const { node, action } = attach('One');
		node.dispatchEvent(new FocusEvent('focusin'));
		action.update('Two');
		expect(bubbleText()).toBe('Two');
		action.destroy();
		expect(bubbleText()).toBeNull();
	});
});

describe('Tooltip wrapper', () => {
	function mountWrapper(props: Record<string, unknown>) {
		const target = document.createElement('div');
		document.body.appendChild(target);
		const component = createClassComponent({
			component: TooltipWrapFixture as never,
			target,
			props
		});
		cleanup.push(() => {
			component.$destroy();
			target.remove();
		});
		return target;
	}

	it('opens on focusin of its trigger and closes on Escape', async () => {
		const target = mountWrapper({ text: 'Hint' });
		const trigger = target.querySelector('button')!.parentElement!;
		trigger.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
		await new Promise((r) => setTimeout(r, 0));
		expect(bubbleText()).toBe('Hint');
		trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
		await new Promise((r) => setTimeout(r, 0));
		expect(bubbleText()).toBeNull();
	});

	it('truncateOnly keeps a non-overflowing trigger silent', async () => {
		const target = mountWrapper({ text: 'Hint', truncateOnly: true });
		const trigger = target.querySelector('button')!.parentElement!;
		trigger.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
		await new Promise((r) => setTimeout(r, 0));
		expect(bubbleText()).toBeNull();
	});
});
