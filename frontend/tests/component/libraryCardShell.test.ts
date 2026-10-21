// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { mount, unmount, flushSync, tick } from 'svelte';
import type { SavedSegment } from '$lib/types/segments';

const { default: SegmentCard } = await import('../../src/routes/prompts/sections/SegmentCard.svelte');

function segment(): SavedSegment {
	return {
		id: 'seg-1',
		name: 'Golden hour',
		category_id: 'cat-light',
		type: 'content',
		content: 'warm light',
		chips: {},
		enabled: true,
		tags: [],
		effective_color: '#F59E0B'
	};
}

let target: HTMLDivElement;
let component: ReturnType<typeof mount> | null = null;

function mountCard(props: Record<string, unknown> = {}) {
	target = document.createElement('div');
	document.body.appendChild(target);
	const handlers = {
		onToggleSelect: vi.fn(),
		onOpen: vi.fn(),
		onInsert: vi.fn(),
		onDuplicate: vi.fn(),
		onDelete: vi.fn()
	};
	component = mount(SegmentCard, { target, props: { segment: segment(), ...handlers, ...props } });
	flushSync();
	return { card: target.querySelector('[data-segment-card]') as HTMLElement, handlers };
}

afterEach(() => {
	if (component) unmount(component);
	component = null;
	target?.remove();
	document.body.innerHTML = '';
});

describe('LibraryCard shell and menu', () => {
	it('exposes the card hook, label and select control', () => {
		const { card } = mountCard();
		expect(card.getAttribute('aria-label')).toBe('Golden hour');
		const toggle = card.querySelector('button[aria-label="Select segment"]') as HTMLButtonElement;
		expect(toggle.getAttribute('aria-pressed')).toBe('false');
	});

	it('toggling selection from the checkbox does not open the card', () => {
		const { card, handlers } = mountCard();
		(card.querySelector('button[aria-label="Select segment"]') as HTMLButtonElement).click();
		expect(handlers.onToggleSelect).toHaveBeenCalledTimes(1);
		expect(handlers.onOpen).not.toHaveBeenCalled();
	});

	it('opening the kebab does not open the card, and picking an item runs its handler only', async () => {
		const { card, handlers } = mountCard();
		(card.querySelector('button[aria-label="More actions"]') as HTMLButtonElement).click();
		await tick();
		flushSync();
		expect(handlers.onOpen).not.toHaveBeenCalled();
		const items = Array.from(document.querySelectorAll('[role="menuitem"]'));
		expect(items.map((i) => i.textContent?.trim())).toEqual(['Duplicate', 'Delete']);
		(items[1] as HTMLElement).click();
		flushSync();
		expect(handlers.onDelete).toHaveBeenCalledTimes(1);
		expect(handlers.onDuplicate).not.toHaveBeenCalled();
		expect(handlers.onOpen).not.toHaveBeenCalled();
	});

	it('Enter on the kebab trigger does not bubble to the card keyboard contract', () => {
		const { card, handlers } = mountCard();
		const trigger = card.querySelector('button[aria-label="More actions"]') as HTMLButtonElement;
		trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
		expect(handlers.onOpen).not.toHaveBeenCalled();
	});

	it('Enter on the card opens it once and prevents the default', () => {
		const { card, handlers } = mountCard();
		const event = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
		card.dispatchEvent(event);
		expect(handlers.onOpen).toHaveBeenCalledTimes(1);
		expect(event.defaultPrevented).toBe(true);
		expect(handlers.onToggleSelect).not.toHaveBeenCalled();
	});

	it('Space on the card toggles selection once and prevents the default', () => {
		const { card, handlers } = mountCard();
		const event = new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true });
		card.dispatchEvent(event);
		expect(handlers.onToggleSelect).toHaveBeenCalledTimes(1);
		expect(event.defaultPrevented).toBe(true);
		expect(handlers.onOpen).not.toHaveBeenCalled();
	});

	it('Enter and Space on inner controls leave the card contract alone', () => {
		const { card, handlers } = mountCard();
		const checkbox = card.querySelector('button[aria-label="Select segment"]') as HTMLElement;
		const insert = Array.from(card.querySelectorAll('button')).find(
			(b) => b.textContent?.trim() === 'Insert'
		) as HTMLElement;
		for (const control of [checkbox, insert]) {
			for (const key of ['Enter', ' ']) {
				const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
				control.dispatchEvent(event);
				expect(event.defaultPrevented).toBe(false);
			}
		}
		expect(handlers.onOpen).not.toHaveBeenCalled();
		expect(handlers.onToggleSelect).not.toHaveBeenCalled();
	});

	it('a selected card reports Deselect and aria-pressed', () => {
		const { card } = mountCard({ selected: true });
		const toggle = card.querySelector('button[aria-label="Deselect segment"]') as HTMLButtonElement;
		expect(toggle.getAttribute('aria-pressed')).toBe('true');
		expect(card.querySelector('button[aria-label="Select segment"]')).toBeNull();
	});

	it('Escape closes the kebab and returns focus to the trigger', async () => {
		const { card } = mountCard();
		const trigger = card.querySelector('button[aria-label="More actions"]') as HTMLButtonElement;
		trigger.click();
		await tick();
		flushSync();
		const menu = document.querySelector('[role="menu"]') as HTMLElement;
		expect(menu).toBeTruthy();
		menu.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
		await tick();
		await tick();
		flushSync();
		expect(document.querySelector('[role="menu"]')).toBeNull();
		expect(document.activeElement).toBe(trigger);
	});
});
