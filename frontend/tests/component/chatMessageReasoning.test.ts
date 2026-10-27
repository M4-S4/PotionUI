import { describe, it, expect, afterEach } from 'vitest';
import { flushSync } from 'svelte';

const { default: ChatMessage } = await import('../../src/lib/components/ChatMessage.svelte');
const { default: ChatBehaviorTrace } = await import(
	'../../src/lib/components/chat/ChatBehaviorTrace.svelte'
);
const { createClassComponent } = await import('svelte/legacy');

function mountComponent(component: unknown, props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	createClassComponent({ component: component as never, target, props });
	flushSync();
	return target;
}

afterEach(() => {
	document.body.innerHTML = '';
});

describe('ChatMessage reasoning block', () => {
	it('puts the reasoning block above the reply text', () => {
		const target = mountComponent(ChatMessage, {
			role: 'assistant',
			content: 'the answer',
			reasoning: [{ kind: 'raw', text: 'weighing options', duration_ms: 4000 }]
		});

		const block = target.querySelector('[data-testid="reasoning-block"]') as HTMLElement;
		const copy = target.querySelector('.assistant-copy') as HTMLElement;
		expect(block).not.toBeNull();
		expect(block.compareDocumentPosition(copy) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(block.textContent).toContain('Thought for 4s');
		expect(copy.textContent).not.toContain('weighing options');
	});

	it('shows a summary and an omitted marker with their own wording', () => {
		const summary = mountComponent(ChatMessage, {
			role: 'assistant',
			content: 'the answer',
			reasoning: [{ kind: 'summary', text: 'Checked both tabs.', duration_ms: 1840 }]
		});
		const omitted = mountComponent(ChatMessage, {
			role: 'assistant',
			content: 'the answer',
			reasoning: [{ kind: 'omitted', text: '', duration_ms: 2000 }]
		});

		expect(summary.querySelector('[data-testid="reasoning-block"]')?.textContent).toContain('Reasoned for 2s');
		expect(omitted.querySelector('[data-testid="reasoning-block"]')?.textContent).toContain('Reasoned for 2s');
		expect(omitted.querySelector('[data-testid="reasoning-toggle"]')).toBeNull();
	});

	it('renders no block when the message has no reasoning', () => {
		const target = mountComponent(ChatMessage, { role: 'assistant', content: 'the answer' });

		expect(target.querySelector('[data-testid="reasoning-block"]')).toBeNull();
	});

	it('streams a live block open beside the partial reply', () => {
		const target = mountComponent(ChatMessage, {
			role: 'assistant',
			content: '',
			isStreaming: true,
			reasoning: [{ kind: 'raw', text: 'reasoning so far', started_at: Date.now(), open: true }]
		});

		expect(target.querySelector('[data-testid="reasoning-block"]')?.getAttribute('data-state')).toBe('live');
		expect(target.querySelector('[data-testid="reasoning-body"]')?.textContent).toContain('reasoning so far');
	});

	it('never shows reasoning on a user message', () => {
		const target = mountComponent(ChatMessage, {
			role: 'user',
			content: 'hello',
			reasoning: [{ kind: 'raw', text: 'x', duration_ms: 1 }]
		});

		expect(target.querySelector('[data-testid="reasoning-block"]')).toBeNull();
	});
});

describe('ChatMessage notices', () => {
	it('shows each notice above the reasoning block and the reply', () => {
		const target = mountComponent(ChatMessage, {
			role: 'assistant',
			content: 'the answer',
			reasoning: [{ kind: 'raw', text: 'plan', duration_ms: 1000 }],
			notices: [{ kind: 'reasoning_dropped', message: 'Reasoning made with a different API key was left out.' }]
		});

		const notice = target.querySelector('[data-testid="chat-notice"]') as HTMLElement;
		const block = target.querySelector('[data-testid="reasoning-block"]') as HTMLElement;
		const copy = target.querySelector('.assistant-copy') as HTMLElement;
		expect(notice.textContent).toContain('Reasoning made with a different API key was left out.');
		expect(notice.compareDocumentPosition(block) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(notice.compareDocumentPosition(copy) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
	});

	it('renders nothing without notices', () => {
		const target = mountComponent(ChatMessage, { role: 'assistant', content: 'the answer' });

		expect(target.querySelector('[data-testid="chat-notices"]')).toBeNull();
	});

	it('never shows notices on a user message', () => {
		const target = mountComponent(ChatMessage, {
			role: 'user',
			content: 'hello',
			notices: [{ kind: 'retrying', message: 'x' }]
		});

		expect(target.querySelector('[data-testid="chat-notice"]')).toBeNull();
	});
});

describe('behavior trace thinking step label', () => {
	const liveStep = (detail?: Record<string, unknown>) => [
		{ step: 'thinking', state: 'started', detail, seq: 0 }
	];

	it('keeps "Thinking…" while model thinking is not shown', () => {
		const target = mountComponent(ChatBehaviorTrace, {
			traceSteps: liveStep(),
			isStreaming: true,
			executions: []
		});

		expect(target.textContent).toContain('Thinking…');
		expect(target.textContent).not.toContain('Waiting for the model…');
	});

	it('says "Waiting for the model…" when model thinking is shown', () => {
		const target = mountComponent(ChatBehaviorTrace, {
			traceSteps: liveStep({ thinking_visible: true }),
			isStreaming: true,
			executions: []
		});

		expect(target.textContent).toContain('Waiting for the model…');
		expect(target.textContent).not.toContain('Thinking…');
	});

	it('reads the same marker from a persisted manifest on reload', () => {
		const target = mountComponent(ChatBehaviorTrace, {
			traceSteps: [],
			executions: [],
			metadata: {
				behavior_trace: {
					steps: [{ step: 'thinking', duration_ms: 12 }],
					thinking_visible: true,
					token_counts: { prompt: null, completion: null }
				}
			}
		});
		(target.querySelector('.behavior-trace-toggle') as HTMLButtonElement).click();
		flushSync();

		expect(target.textContent).toContain('Waiting for the model…');
	});

	it('keeps the old wording for a persisted manifest without the marker', () => {
		const target = mountComponent(ChatBehaviorTrace, {
			traceSteps: [],
			executions: [],
			metadata: {
				behavior_trace: {
					steps: [{ step: 'thinking', duration_ms: 12 }],
					token_counts: { prompt: null, completion: null }
				}
			}
		});
		(target.querySelector('.behavior-trace-toggle') as HTMLButtonElement).click();
		flushSync();

		expect(target.textContent).toContain('Thinking…');
	});
});
