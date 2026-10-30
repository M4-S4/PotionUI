// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { keyboardFocus, pointerFocus } from './helpers/keyboardFocus';

const { default: tooltip } = await import('$lib/actions/tooltip');
const { default: TooltipWrapFixture } = await import('./fixtures/TooltipWrapFixture.svelte');
const { createClassComponent } = await import('svelte/legacy');
const { flushSync } = await import('svelte');

let cleanup: Array<() => void> = [];

function attach(text = 'Hint', tag = 'button') {
	const node = document.createElement(tag);
	node.textContent = 'x';
	document.body.appendChild(node);
	const action = tooltip(node, text);
	cleanup.push(() => {
		action.destroy();
		node.remove();
	});
	return { node, action };
}

const bubble = () => document.body.querySelector('[role="tooltip"]');

async function openByHover(node: HTMLElement) {
	node.dispatchEvent(new MouseEvent('mouseenter'));
	vi.advanceTimersByTime(250);
	await Promise.resolve();
}

afterEach(() => {
	cleanup.forEach((fn) => fn());
	cleanup = [];
	vi.useRealTimers();
});

describe('tooltip trigger focus', () => {
	it('does not open for focus that is not keyboard-driven', async () => {
		const { node } = attach();
		pointerFocus(node);
		await Promise.resolve();
		expect(bubble()).toBeNull();
	});

	it('opens for keyboard focus', async () => {
		const { node } = attach();
		keyboardFocus(node);
		await Promise.resolve();
		expect(bubble()).not.toBeNull();
	});

	it('ignores focus restored while the window was inactive', async () => {
		const { node } = attach();
		window.dispatchEvent(new Event('blur'));
		keyboardFocus(node);
		await Promise.resolve();
		expect(bubble()).toBeNull();
		window.dispatchEvent(new Event('focus'));
	});

	it('ignores focus restored right after the window regains focus', async () => {
		const { node } = attach();
		window.dispatchEvent(new Event('blur'));
		window.dispatchEvent(new Event('focus'));
		keyboardFocus(node);
		await Promise.resolve();
		expect(bubble()).toBeNull();
		await new Promise((r) => setTimeout(r, 300));
		node.blur();
		keyboardFocus(node);
		await Promise.resolve();
		expect(bubble()).not.toBeNull();
	});
});

describe('tooltip trigger dismissal', () => {
	it('hides when the window loses focus', async () => {
		vi.useFakeTimers();
		const { node } = attach();
		await openByHover(node);
		expect(bubble()).not.toBeNull();
		window.dispatchEvent(new Event('blur'));
		expect(bubble()).toBeNull();
		window.dispatchEvent(new Event('focus'));
	});

	it('hides when the document becomes hidden', async () => {
		vi.useFakeTimers();
		const { node } = attach();
		await openByHover(node);
		const spy = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden');
		document.dispatchEvent(new Event('visibilitychange'));
		spy.mockRestore();
		expect(bubble()).toBeNull();
	});

	it('does not start a hover while the document is hidden', async () => {
		vi.useFakeTimers();
		const { node } = attach();
		const spy = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden');
		await openByHover(node);
		spy.mockRestore();
		expect(bubble()).toBeNull();
	});

	it('hides when the anchor is removed from the DOM while open', async () => {
		vi.useFakeTimers();
		const { node } = attach();
		await openByHover(node);
		expect(bubble()).not.toBeNull();
		node.remove();
		await Promise.resolve();
		expect(bubble()).toBeNull();
	});

	it('hides when the pointer moves outside the anchor without a mouseleave', async () => {
		vi.useFakeTimers();
		const { node } = attach();
		await openByHover(node);
		document.body.dispatchEvent(new Event('pointermove', { bubbles: true }));
		expect(bubble()).toBeNull();
	});

	it('hides on pointerdown, scroll of an ancestor and Escape', async () => {
		vi.useFakeTimers();
		const { node } = attach();
		for (const fire of [
			() => document.body.dispatchEvent(new Event('pointerdown', { bubbles: true })),
			() => document.dispatchEvent(new Event('scroll')),
			() => document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
		]) {
			await openByHover(node);
			expect(bubble()).not.toBeNull();
			fire();
			expect(bubble()).toBeNull();
		}
	});

	it('destroy clears a pending show timer', async () => {
		vi.useFakeTimers();
		const { node, action } = attach();
		node.dispatchEvent(new MouseEvent('mouseenter'));
		action.destroy();
		vi.advanceTimersByTime(500);
		await Promise.resolve();
		expect(bubble()).toBeNull();
	});

	it('does not show a pending tooltip for an anchor removed before the delay', async () => {
		vi.useFakeTimers();
		const { node } = attach();
		node.dispatchEvent(new MouseEvent('mouseenter'));
		node.remove();
		vi.advanceTimersByTime(500);
		await Promise.resolve();
		expect(bubble()).toBeNull();
	});
});

describe('Tooltip component dismissal', () => {
	function mountTooltip() {
		const target = document.createElement('div');
		document.body.appendChild(target);
		const component = createClassComponent({
			component: TooltipWrapFixture as never,
			target,
			props: { text: 'Chip' }
		});
		cleanup.push(() => {
			component.$destroy();
			target.remove();
		});
		return target;
	}

	it('hides on window blur and when its trigger is detached', async () => {
		vi.useFakeTimers();
		const target = mountTooltip();
		const trigger = target.querySelector('button')!.parentElement as HTMLElement;
		await openByHover(trigger);
		expect(bubble()?.textContent?.trim()).toBe('Chip');
		window.dispatchEvent(new Event('blur'));
		flushSync();
		expect(bubble()).toBeNull();
		window.dispatchEvent(new Event('focus'));
		await openByHover(trigger);
		expect(bubble()).not.toBeNull();
		trigger.remove();
		await Promise.resolve();
		flushSync();
		expect(bubble()).toBeNull();
	});
});
