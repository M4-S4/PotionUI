import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';
import type { ReasoningBlock } from '../../src/lib/types/chat';

const { default: ChatReasoningBlock } = await import(
	'../../src/lib/components/chat/ChatReasoningBlock.svelte'
);
const { createClassComponent } = await import('svelte/legacy');

interface Props {
	blocks: ReasoningBlock[];
	isStreaming: boolean;
}

let instances: Array<{ $destroy: () => void }> = [];

function mount(props: Props) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({ component: ChatReasoningBlock as never, target, props });
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
	return target.querySelector('[data-testid="reasoning-block"]') as HTMLElement | null;
}

function toggle(target: HTMLElement) {
	return target.querySelector('[data-testid="reasoning-toggle"]') as HTMLButtonElement;
}

function body(target: HTMLElement) {
	return target.querySelector('[data-testid="reasoning-body"]') as HTMLElement | null;
}

function label(target: HTMLElement) {
	return target.querySelector('[data-testid="reasoning-label"]')?.textContent?.trim();
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

const live = (text: string, startedAgoMs = 0, kind: ReasoningBlock['kind'] = 'raw'): ReasoningBlock => ({
	kind,
	text,
	started_at: Date.now() - startedAgoMs,
	open: true
});

const done = (text: string, duration_ms?: number, kind: ReasoningBlock['kind'] = 'raw'): ReasoningBlock => ({
	kind,
	text,
	duration_ms
});

const omitted = (duration_ms: number | null = null): ReasoningBlock => ({ kind: 'omitted', text: '', duration_ms });

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

describe('ChatReasoningBlock', () => {
	it('renders nothing without reasoning', () => {
		expect(block(mount({ blocks: [], isStreaming: false }).target)).toBeNull();
		expect(block(mount({ blocks: [done('   \n')], isStreaming: false }).target)).toBeNull();
	});

	it('shows a live block open while a raw block is streaming', () => {
		const { target } = mount({ blocks: [live('weighing options', 3000)], isStreaming: true });

		expect(block(target)?.dataset.state).toBe('live');
		expect(block(target)?.dataset.kind).toBe('raw');
		expect(label(target)).toBe('Thinking…');
		expect(toggle(target).getAttribute('aria-expanded')).toBe('true');
		expect(body(target)?.textContent).toContain('weighing options');
		expect(target.querySelector('[data-testid="reasoning-elapsed"]')?.textContent).toBe('3s');
	});

	it('keeps the elapsed time moving while live', () => {
		const { target } = mount({ blocks: [live('hm', 5000)], isStreaming: true });
		const elapsed = () => target.querySelector('[data-testid="reasoning-elapsed"]')?.textContent;
		expect(elapsed()).toBe('5s');

		vi.advanceTimersByTime(2000);
		flushSync();

		expect(elapsed()).toBe('7s');
	});

	it('folds itself and reports the time once the block closes', async () => {
		const { target, update } = mount({ blocks: [live('hm', 2000)], isStreaming: true });

		await update({ blocks: [done('hm', 12000)] });

		expect(block(target)?.dataset.state).toBe('done');
		expect(label(target)).toBe('Thought for 12s');
		expect(toggle(target).getAttribute('aria-expanded')).toBe('false');
		expect(body(target)).toBeNull();
		expect(target.querySelector('[data-testid="reasoning-elapsed"]')).toBeNull();
	});

	it('can be reopened after folding and closed again', async () => {
		const { target } = mount({ blocks: [done('the full reasoning', 4000)], isStreaming: false });

		await click(target);
		expect(body(target)?.textContent).toContain('the full reasoning');
		expect(toggle(target).getAttribute('aria-expanded')).toBe('true');

		await click(target);
		expect(body(target)).toBeNull();
	});

	it('is collapsed when a finished message is reopened', () => {
		const { target } = mount({ blocks: [done('old reasoning', 65000)], isStreaming: false });

		expect(body(target)).toBeNull();
		expect(label(target)).toBe('Thought for 1m 05s');
	});

	it('says only "Thought" when no duration is known', () => {
		const { target } = mount({ blocks: [done('old reasoning')], isStreaming: false });

		expect(label(target)).toBe('Thought');
	});

	it('sums the time of every block and shows each in order', async () => {
		const { target } = mount({
			blocks: [done('first turn', 3000), done('second turn', 4000)],
			isStreaming: false
		});

		expect(label(target)).toBe('Thought for 7s');
		await click(target);
		const texts = [...target.querySelectorAll('[data-testid="reasoning-segment"]')].map((el) =>
			el.textContent?.trim()
		);
		expect(texts).toEqual(['first turn', 'second turn']);
	});

	it('opens again for the next turn of a tool round', async () => {
		const { target, update } = mount({ blocks: [live('plan', 1000)], isStreaming: true });
		await update({ blocks: [done('plan', 1000)] });
		expect(body(target)).toBeNull();

		await update({ blocks: [done('plan', 1000), live('second look')], isStreaming: true });

		expect(block(target)?.dataset.state).toBe('live');
		expect(body(target)?.textContent).toContain('second look');
	});

	it('respects a manual fold while still streaming', async () => {
		const { target, update } = mount({ blocks: [live('plan', 1000)], isStreaming: true });

		await click(target);
		expect(body(target)).toBeNull();

		await update({ blocks: [live('plan and more', 1500)] });
		expect(body(target)).toBeNull();
		expect(label(target)).toBe('Thinking…');
	});

	it('respects a manual open that outlives the live phase', async () => {
		const { target, update } = mount({ blocks: [done('plan', 1000)], isStreaming: false });
		await click(target);

		await update({ blocks: [done('plan', 1000), done('more', 500)] });

		expect(body(target)).not.toBeNull();
	});

	it('does not treat a stale open flag as live once streaming stopped', () => {
		const { target } = mount({ blocks: [live('cut off')], isStreaming: false });

		expect(block(target)?.dataset.state).toBe('done');
		expect(body(target)).toBeNull();
	});

	it('renders the text verbatim, never as markup', () => {
		const { target } = mount({ blocks: [live('<b>bold</b> **x**')], isStreaming: true });

		expect(body(target)?.querySelector('b')).toBeNull();
		expect(body(target)?.textContent).toContain('<b>bold</b> **x**');
	});

	it('uses no native title tooltip', () => {
		const { target } = mount({ blocks: [live('hm')], isStreaming: true });

		expect(target.querySelector('[title]')).toBeNull();
	});

	it('exposes the reasoning as a labelled region', () => {
		const { target } = mount({ blocks: [live('hm')], isStreaming: true });

		expect(body(target)?.getAttribute('role')).toBe('region');
		expect(body(target)?.getAttribute('aria-label')).toBe('Model reasoning');
	});
});

describe('ChatReasoningBlock summary blocks', () => {
	it('words a finished summary as "Reasoned" and labels its text as a summary once opened', async () => {
		const { target } = mount({
			blocks: [done('Checked both tabs.', 1840, 'summary')],
			isStreaming: false
		});

		expect(block(target)?.dataset.kind).toBe('summary');
		expect(label(target)).toBe('Reasoned for 2s');
		expect(body(target)).toBeNull();

		await click(target);
		expect(body(target)?.textContent).toContain('Checked both tabs.');
		expect(body(target)?.textContent).toContain('Summary');
	});

	it('shows no summary label on raw reasoning', async () => {
		const { target } = mount({ blocks: [done('raw thoughts', 1000)], isStreaming: false });

		await click(target);
		expect(body(target)?.textContent).not.toContain('Summary');
	});

	it('says "Reasoning…" while a summary streams', () => {
		const { target } = mount({ blocks: [live('Checking', 500, 'summary')], isStreaming: true });

		expect(label(target)).toBe('Reasoning…');
		expect(toggle(target).getAttribute('aria-expanded')).toBe('true');
	});

	it('is collapsed by default once the stream is over', () => {
		const { target } = mount({
			blocks: [done('Checked both tabs.', 900, 'summary')],
			isStreaming: false
		});

		expect(toggle(target).getAttribute('aria-expanded')).toBe('false');
	});
});

describe('ChatReasoningBlock omitted blocks', () => {
	it('is a quiet marker with the duration and no text, toggle or body', () => {
		const { target } = mount({ blocks: [omitted(4200)], isStreaming: false });

		expect(block(target)?.dataset.kind).toBe('omitted');
		expect(label(target)).toBe('Reasoned for 4s');
		expect(target.querySelector('button')).toBeNull();
		expect(body(target)).toBeNull();
	});

	it('says only "Reasoned" when the duration is unknown', () => {
		const { target } = mount({ blocks: [omitted()], isStreaming: false });

		expect(label(target)).toBe('Reasoned');
	});

	it('sums several omitted blocks', () => {
		const { target } = mount({ blocks: [omitted(1000), omitted(2000)], isStreaming: false });

		expect(label(target)).toBe('Reasoned for 3s');
	});

	it('shows a spinner marker while an omitted block is still open', () => {
		const { target } = mount({
			blocks: [{ kind: 'omitted', text: '', started_at: Date.now(), open: true }],
			isStreaming: true
		});

		expect(block(target)?.dataset.state).toBe('live');
		expect(label(target)).toBe('Reasoning…');
		expect(target.querySelector('[role="status"]')).not.toBeNull();
	});

	it('notes an omitted block between text blocks inside the opened body', async () => {
		const { target } = mount({
			blocks: [done('first', 1000), omitted(2000), done('third', 1000)],
			isStreaming: false
		});

		expect(block(target)?.dataset.kind).toBe('mixed');
		await click(target);
		const segments = [...target.querySelectorAll('[data-testid="reasoning-segment"]')];
		expect(segments.map((el) => el.getAttribute('data-kind'))).toEqual(['raw', 'omitted', 'raw']);
		expect(segments[1].textContent).toContain('Reasoning not shown');
		expect(segments[1].textContent).toContain('2s');
	});
});
