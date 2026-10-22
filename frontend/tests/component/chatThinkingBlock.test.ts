import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';
import type { ThinkingSegment } from '../../src/lib/types/chat';

const { default: ChatThinkingBlock } = await import(
	'../../src/lib/components/chat/ChatThinkingBlock.svelte'
);
const { createClassComponent } = await import('svelte/legacy');

interface Props {
	segments: ThinkingSegment[];
	isStreaming: boolean;
}

let instances: Array<{ $destroy: () => void }> = [];

function mount(props: Props) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({ component: ChatThinkingBlock as never, target, props });
	instances.push(component);
	return {
		target,
		async update(next: Partial<Props>) {
			component.$set(next);
			flushSync();
			await settle();
		}
	};
}

function block(target: HTMLElement) {
	return target.querySelector('[data-testid="thinking-block"]') as HTMLElement | null;
}

function toggle(target: HTMLElement) {
	return target.querySelector('[data-testid="thinking-toggle"]') as HTMLButtonElement;
}

function body(target: HTMLElement) {
	return target.querySelector('[data-testid="thinking-body"]') as HTMLElement | null;
}

async function settle() {
	for (let i = 0; i < 3; i++) await Promise.resolve();
	flushSync();
}

async function click(target: HTMLElement) {
	toggle(target).click();
	flushSync();
	await settle();
}

const live = (text: string, startedAgoMs = 0): ThinkingSegment => ({
	text,
	started_at: Date.now() - startedAgoMs,
	open: true
});

const done = (text: string, duration_ms?: number): ThinkingSegment => ({ text, duration_ms });

beforeEach(() => {
	vi.useFakeTimers();
	vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));
});

afterEach(() => {
	for (const instance of instances) instance.$destroy();
	instances = [];
	document.body.innerHTML = '';
	vi.useRealTimers();
});

describe('ChatThinkingBlock', () => {
	it('renders nothing without thinking text', () => {
		expect(block(mount({ segments: [], isStreaming: false }).target)).toBeNull();
		expect(block(mount({ segments: [done('   \n')], isStreaming: false }).target)).toBeNull();
	});

	it('shows a live block open while a segment is streaming', () => {
		const { target } = mount({ segments: [live('weighing options', 3000)], isStreaming: true });

		expect(block(target)?.dataset.state).toBe('live');
		expect(toggle(target).textContent).toContain('Thinking…');
		expect(toggle(target).getAttribute('aria-expanded')).toBe('true');
		expect(body(target)?.textContent).toContain('weighing options');
		expect(target.querySelector('[data-testid="thinking-elapsed"]')?.textContent).toBe('3s');
	});

	it('keeps the elapsed time moving while live', () => {
		const { target } = mount({ segments: [live('hm', 5000)], isStreaming: true });
		const elapsed = () => target.querySelector('[data-testid="thinking-elapsed"]')?.textContent;
		expect(elapsed()).toBe('5s');

		vi.advanceTimersByTime(2000);
		flushSync();

		expect(elapsed()).toBe('7s');
	});

	it('folds itself and reports the time once the segment closes', async () => {
		const { target, update } = mount({ segments: [live('hm', 2000)], isStreaming: true });

		await update({ segments: [done('hm', 12000)] });

		expect(block(target)?.dataset.state).toBe('done');
		expect(toggle(target).textContent).toContain('Thought for 12s');
		expect(toggle(target).getAttribute('aria-expanded')).toBe('false');
		expect(body(target)).toBeNull();
		expect(target.querySelector('[data-testid="thinking-elapsed"]')).toBeNull();
	});

	it('can be reopened after folding and closed again', async () => {
		const { target } = mount({ segments: [done('the full reasoning', 4000)], isStreaming: false });

		await click(target);
		expect(body(target)?.textContent).toContain('the full reasoning');
		expect(toggle(target).getAttribute('aria-expanded')).toBe('true');

		await click(target);
		expect(body(target)).toBeNull();
	});

	it('is collapsed when a finished message is reopened', () => {
		const { target } = mount({ segments: [done('old reasoning', 65000)], isStreaming: false });

		expect(body(target)).toBeNull();
		expect(toggle(target).textContent).toContain('Thought for 1m 05s');
	});

	it('says only "Thought" when no duration is known', () => {
		const { target } = mount({ segments: [done('old reasoning')], isStreaming: false });

		expect(toggle(target).textContent?.trim()).toMatch(/^Thought$/);
	});

	it('sums the time of every turn and shows each turn in order', async () => {
		const { target } = mount({
			segments: [done('first turn', 3000), done('second turn', 4000)],
			isStreaming: false
		});

		expect(toggle(target).textContent).toContain('Thought for 7s');
		await click(target);
		const texts = [...target.querySelectorAll('[data-testid="thinking-segment"]')].map((el) =>
			el.textContent?.trim()
		);
		expect(texts).toEqual(['first turn', 'second turn']);
	});

	it('opens again for the next turn of a tool round', async () => {
		const { target, update } = mount({ segments: [live('plan', 1000)], isStreaming: true });
		await update({ segments: [done('plan', 1000)] });
		expect(body(target)).toBeNull();

		await update({ segments: [done('plan', 1000), live('second look')], isStreaming: true });

		expect(block(target)?.dataset.state).toBe('live');
		expect(body(target)?.textContent).toContain('second look');
	});

	it('respects a manual fold while still streaming', async () => {
		const { target, update } = mount({ segments: [live('plan', 1000)], isStreaming: true });

		await click(target);
		expect(body(target)).toBeNull();

		await update({ segments: [live('plan and more', 1500)] });
		expect(body(target)).toBeNull();
		expect(toggle(target).textContent).toContain('Thinking…');
	});

	it('respects a manual open that outlives the live phase', async () => {
		const { target, update } = mount({ segments: [done('plan', 1000)], isStreaming: false });
		await click(target);

		await update({ segments: [done('plan', 1000), done('more', 500)] });

		expect(body(target)).not.toBeNull();
	});

	it('does not treat a stale open flag as live once streaming stopped', () => {
		const { target } = mount({ segments: [live('cut off')], isStreaming: false });

		expect(block(target)?.dataset.state).toBe('done');
		expect(body(target)).toBeNull();
	});

	it('renders the text verbatim, never as markup', () => {
		const { target } = mount({ segments: [live('<b>bold</b> **x**')], isStreaming: true });

		expect(body(target)?.querySelector('b')).toBeNull();
		expect(body(target)?.textContent).toContain('<b>bold</b> **x**');
	});

	it('uses no native title tooltip', () => {
		const { target } = mount({ segments: [live('hm')], isStreaming: true });

		expect(target.querySelector('[title]')).toBeNull();
	});

	it('exposes the reasoning as a labelled region', () => {
		const { target } = mount({ segments: [live('hm')], isStreaming: true });

		expect(body(target)?.getAttribute('role')).toBe('region');
		expect(body(target)?.getAttribute('aria-label')).toBe('Model thinking');
	});
});
