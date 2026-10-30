// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { settle } from './helpers/settle';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getMyLLMConfigurations: vi.fn().mockResolvedValue({
			success: true,
			data: { llm_configs: [{ id: 'cfg-1', name: 'Test config' }] }
		}),
		getChatModes: vi.fn().mockResolvedValue({ success: true, data: { modes: [] } }),
		listChatTools: vi.fn().mockResolvedValue({ success: true, data: { tools: [] } }),
		getChatSessions: vi.fn().mockResolvedValue({ success: true, data: { sessions: [] } }),
		getMyToolsetPreferences: vi.fn().mockResolvedValue({ success: false }),
		listPresets: vi.fn().mockResolvedValue({ success: false }),
		setOnAuthExpired: vi.fn(),
		setOnTokenRefreshed: vi.fn(),
		clearAuth: vi.fn()
	}
}));

vi.mock('$app/stores', () => {
	const { writable } = require('svelte/store');
	return { page: writable({ url: new URL('http://localhost/generate') }) };
});

const { chatSession } = await import('$lib/stores/chatSession');
const { default: UnifiedAIChat } = await import('$lib/components/UnifiedAIChat.svelte');
const { createClassComponent } = await import('svelte/legacy');

let target: HTMLElement;
let component: { $destroy: () => void } | undefined;

async function open() {
	component = createClassComponent({ component: UnifiedAIChat as never, target, props: {} });
	await settle(10);
}

beforeEach(() => {
	target = document.createElement('div');
	document.body.appendChild(target);
});

afterEach(() => {
	component?.$destroy();
	component = undefined;
	target.remove();
	chatSession.reset();
	vi.clearAllMocks();
});

describe('UnifiedAIChat reasoning and notices', () => {
	it('hands a stored message its reasoning blocks and notices', async () => {
		chatSession.loadedSession({ id: 'sess-1', mode: 'generation' }, [
			{ role: 'user', content: 'which palette?', timestamp: 1 } as never,
			{
				role: 'assistant',
				content: 'the softer one',
				timestamp: 2,
				reasoning: [{ kind: 'summary', text: 'Checked both tabs.', duration_ms: 1840 }],
				notices: [{ kind: 'reasoning_dropped', message: 'Reasoning made with a different API key was left out.' }]
			} as never
		]);
		await open();

		expect(target.querySelector('[data-testid="reasoning-block"]')?.textContent).toContain('Reasoned for 2s');
		expect(target.querySelector('[data-testid="chat-notice"]')?.textContent).toContain('different API key');
	});

	it('shows no reasoning block for a stored message without reasoning', async () => {
		chatSession.loadedSession({ id: 'sess-1', mode: 'generation' }, [
			{ role: 'user', content: 'hi', timestamp: 1 } as never,
			{ role: 'assistant', content: 'hello', timestamp: 2 } as never
		]);
		await open();

		expect(target.querySelector('[data-testid="reasoning-block"]')).toBeNull();
		expect(target.querySelector('[data-testid="chat-notices"]')).toBeNull();
	});

	it('shows a retry notice in the reply while nothing else has streamed yet', async () => {
		chatSession.loadedSession({ id: 'sess-1', mode: 'generation' }, [
			{ role: 'user', content: 'hi', timestamp: 1 } as never,
			{ role: 'assistant', content: '', timestamp: 2, isStreaming: true } as never
		]);
		await open();
		expect(target.querySelector('.thinking-dots')).not.toBeNull();

		chatSession.applyStreamEvent({
			type: 'notice',
			data: { kind: 'retrying', message: 'Rate limited; retrying in 4 s.' }
		});
		await settle();

		expect(target.querySelector('.thinking-dots')).toBeNull();
		expect(target.querySelector('[data-testid="chat-notice"]')?.textContent).toContain('retrying in 4 s');
	});

	it('streams a live reasoning block open into the reply', async () => {
		chatSession.loadedSession({ id: 'sess-1', mode: 'generation' }, [
			{ role: 'user', content: 'hi', timestamp: 1 } as never,
			{ role: 'assistant', content: '', timestamp: 2, isStreaming: true } as never
		]);
		await open();

		chatSession.applyStreamEvent({ type: 'reasoning', data: { block: 0, kind: 'raw' } }, { accumulated: 'weighing' });
		await settle();

		const block = target.querySelector('[data-testid="reasoning-block"]');
		expect(block?.getAttribute('data-state')).toBe('live');
		expect(target.querySelector('[data-testid="reasoning-body"]')?.textContent).toContain('weighing');
	});
});
