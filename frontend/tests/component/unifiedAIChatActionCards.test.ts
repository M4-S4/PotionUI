import { describe, it, expect, vi, afterEach } from 'vitest';

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
		clearAuth: vi.fn(),
		listMemory: vi.fn().mockResolvedValue({
			success: true,
			data: { notes: [], injection: { cap_per_group: 0, max_content_len: 0 } }
		}),
		resolveToolExecution: vi.fn(),
		sendChatMessageStream: vi.fn(),
		reattachChatMessageStream: vi.fn(),
		getChatSession: vi.fn().mockResolvedValue({ success: false })
	}
}));

vi.mock('$app/stores', () => {
	const { writable } = require('svelte/store');
	return { page: writable({ url: new URL('http://localhost/generate') }) };
});

class StubResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}

class FakeIntersectionObserver {
	static instances: FakeIntersectionObserver[] = [];
	nodes: Element[] = [];
	disconnected = false;
	constructor(public cb: IntersectionObserverCallback) {
		FakeIntersectionObserver.instances.push(this);
	}
	observe(node: Element) {
		this.nodes.push(node);
	}
	unobserve() {}
	disconnect() {
		this.disconnected = true;
	}
	fire(isIntersecting: boolean) {
		this.cb(this.nodes.map((target) => ({ target, isIntersecting }) as IntersectionObserverEntry), this as never);
	}
}

const { api } = await import('$lib/services/api/index');
const { chatSession } = await import('$lib/stores/chatSession');
const { chatComposerDrafts } = await import('$lib/stores/chatComposerDrafts');
const { default: UnifiedAIChat } = await import('$lib/components/UnifiedAIChat.svelte');
const { createClassComponent } = await import('svelte/legacy');
const { get } = await import('svelte/store');

const pendingExecution = {
	tool_name: 'update_form_settings',
	arguments: {},
	result: {
		success: true,
		data: JSON.stringify({ proposed_changes: [{ field_name: 'steps', old_value: 36, new_value: 24 }] })
	},
	duration_ms: 0,
	approval_status: 'pending' as const
};

async function settle() {
	for (let i = 0; i < 12; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

let originalIO: unknown;
let originalRO: unknown;
let target: HTMLElement;
let component: { $destroy: () => void } | undefined;

async function mountWithPendingCard() {
	originalIO = (globalThis as any).IntersectionObserver;
	originalRO = (globalThis as any).ResizeObserver;
	(globalThis as any).IntersectionObserver = FakeIntersectionObserver;
	(globalThis as any).ResizeObserver = StubResizeObserver;
	FakeIntersectionObserver.instances = [];
	chatSession.loadedSession(
		{ id: 'sess-1', mode: 'generation' },
		[
			{ id: 'u1', role: 'user', content: 'fewer steps', timestamp: 1 },
			{ id: 'a1', role: 'assistant', content: 'Fewer steps keeps detail.', timestamp: 2, tool_executions: [pendingExecution] }
		] as never,
		{ messageCount: 2, hasEarlier: false }
	);
	target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({ component: UnifiedAIChat as never, target, props: {} });
	await settle();
}

function card(): HTMLElement | null {
	return target.querySelector('[data-testid="action-card"]');
}

function cardObserver(): FakeIntersectionObserver | undefined {
	return FakeIntersectionObserver.instances.find((o) => o.nodes.includes(card()!));
}

function streamReply(text: string, extra: Record<string, unknown> = {}) {
	vi.mocked(api.sendChatMessageStream).mockImplementation(async (_sid, _payload, onEvent) => {
		await onEvent!({ type: 'token', data: { content: text } });
		await onEvent!({ type: 'done', data: { assistant_message: { id: 'a2', content: text }, ...extra } });
	});
}

afterEach(() => {
	component?.$destroy();
	component = undefined;
	target?.remove();
	chatSession.reset();
	chatComposerDrafts.clear('sess-1');
	(globalThis as any).IntersectionObserver = originalIO;
	(globalThis as any).ResizeObserver = originalRO;
	vi.clearAllMocks();
});

describe('UnifiedAIChat inline action cards', () => {
	it('renders the pending card in the transcript and leaves the composer unlocked', async () => {
		await mountWithPendingCard();
		expect(card()?.dataset.state).toBe('pending');
		expect(target.querySelector('.question-dock')).toBeNull();
		const textbox = target.querySelector('[role="textbox"][aria-placeholder]') as HTMLElement;
		expect(textbox.getAttribute('contenteditable')).toBe('true');
		expect(textbox.getAttribute('aria-placeholder')).toBe('Reply, or tell it what to change instead…');
	});

	it('approve resolves with the current context, then streams the continuation', async () => {
		await mountWithPendingCard();
		vi.mocked(api.resolveToolExecution).mockResolvedValue({
			success: true,
			data: {
				status: 'approved',
				message_id: 'a1',
				tool_index: 0,
				execution: {
					...pendingExecution,
					approval_status: 'approved',
					result: {
						success: true,
						data: JSON.stringify({
							action: 'apply_form_changes',
							applied_changes: [{ field_name: 'steps', old_value: 36, new_value: 24 }]
						})
					}
				}
			}
		});
		streamReply('Done. The form is set to 24 steps.');
		const toolApproved = vi.fn();
		window.addEventListener('potionui:tool-approved', toolApproved);

		(card()!.querySelector('.action-btn.is-approve') as HTMLButtonElement).click();
		await settle();
		window.removeEventListener('potionui:tool-approved', toolApproved);

		expect(api.resolveToolExecution).toHaveBeenCalledWith(
			'sess-1',
			expect.objectContaining({
				message_id: 'a1',
				tool_index: 0,
				approved: true,
				context_metadata: expect.objectContaining({ form_state: expect.any(Object), segments: expect.any(Array) })
			})
		);
		expect(toolApproved).toHaveBeenCalledTimes(1);
		expect(api.sendChatMessageStream).toHaveBeenCalledWith(
			'sess-1',
			expect.objectContaining({ content: '', continueFrom: { message_id: 'a1', tool_index: 0 } }),
			expect.any(Function)
		);
		const state = get(chatSession);
		expect(state.messages.map((m) => m.id)).toEqual(['u1', 'a1', 'a2']);
		expect(state.messages[1].tool_executions![0].approval_status).toBe('approved');
		expect(state.messages[2].content).toBe('Done. The form is set to 24 steps.');
		expect(card()?.dataset.state).toBe('applied');
		expect(target.textContent).toContain('Done. The form is set to 24 steps.');
	});

	it('a refused continuation shows its error on the card, not in the banner', async () => {
		await mountWithPendingCard();
		vi.mocked(api.resolveToolExecution).mockResolvedValue({
			success: true,
			data: {
				status: 'rejected',
				message_id: 'a1',
				tool_index: 0,
				execution: { ...pendingExecution, approval_status: 'rejected' }
			}
		});
		vi.mocked(api.sendChatMessageStream).mockRejectedValue(
			Object.assign(new Error('The message has already been continued.'), {
				httpStatus: 409,
				code: 'continuation_conflict'
			})
		);

		(card()!.querySelector('.action-btn.is-reject') as HTMLButtonElement).click();
		await settle();
		const confirm = card()!.querySelectorAll('.action-reason .action-btn.is-reject')[0] as HTMLButtonElement;
		confirm.click();
		await settle();

		expect(card()?.dataset.state).toBe('rejected');
		expect(card()?.querySelector('[role="alert"]')?.textContent).toBe('The message has already been continued.');
		expect(target.querySelector('.chat-error-banner')).toBeNull();
		expect(get(chatSession).messages.map((m) => m.id)).toEqual(['u1', 'a1']);
	});

	it('a stale resolve keeps the card pending in the form-changed state and does not continue', async () => {
		await mountWithPendingCard();
		vi.mocked(api.resolveToolExecution).mockResolvedValue({
			success: true,
			data: { status: 'stale', message_id: 'a1', tool_index: 0, execution: pendingExecution }
		});

		(card()!.querySelector('.action-btn.is-approve') as HTMLButtonElement).click();
		await settle();

		expect(card()?.dataset.state).toBe('stale');
		expect(api.sendChatMessageStream).not.toHaveBeenCalled();
	});

	it('already_resolved updates the record without a continuation', async () => {
		await mountWithPendingCard();
		vi.mocked(api.resolveToolExecution).mockResolvedValue({
			success: true,
			data: {
				status: 'already_resolved',
				message_id: 'a1',
				tool_index: 0,
				execution: { ...pendingExecution, approval_status: 'rejected', approval_reason: 'earlier' }
			}
		});

		(card()!.querySelector('.action-btn.is-approve') as HTMLButtonElement).click();
		await settle();

		expect(card()?.dataset.state).toBe('rejected');
		expect(api.sendChatMessageStream).not.toHaveBeenCalled();
	});

	it('sending a message while a card is pending rejects it with the message as the reason', async () => {
		chatComposerDrafts.save('sess-1', { text: 'make it blue instead', resources: {} });
		await mountWithPendingCard();
		let rejectedBeforeStream: string | null | undefined;
		vi.mocked(api.sendChatMessageStream).mockImplementation(async (_sid, _payload, onEvent) => {
			rejectedBeforeStream = get(chatSession).messages[1].tool_executions![0].approval_reason;
			await onEvent!({
				type: 'done',
				data: {
					user_message: { id: 'u2' },
					assistant_message: { id: 'a2', content: 'Blue it is.' },
					resolved_message: {
						id: 'a1',
						tool_executions: [{ ...pendingExecution, approval_status: 'rejected', approval_reason: 'make it blue instead' }]
					}
				}
			});
		});

		(target.querySelector('button[aria-label="Send message"]') as HTMLButtonElement).click();
		await settle();

		expect(rejectedBeforeStream).toBe('make it blue instead');
		expect(api.sendChatMessageStream).toHaveBeenCalledWith(
			'sess-1',
			expect.not.objectContaining({ continueFrom: expect.anything() }),
			expect.any(Function)
		);
		expect(api.resolveToolExecution).not.toHaveBeenCalled();
		const state = get(chatSession);
		expect(state.messages.map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2']);
		expect(state.messages[1].tool_executions![0]).toMatchObject({
			approval_status: 'rejected',
			approval_reason: 'make it blue instead'
		});
		expect(card()?.dataset.state).toBe('rejected');
	});

	it('shows the jump pill only while the pending card is out of view, and it scrolls to the card', async () => {
		await mountWithPendingCard();
		expect(target.querySelector('[data-testid="action-jump"]')).toBeNull();
		const observer = cardObserver();
		expect(observer).toBeTruthy();

		observer!.fire(false);
		await settle();
		const pill = target.querySelector('[data-testid="action-jump"]') as HTMLButtonElement;
		expect(pill?.textContent).toContain('1 action waiting for approval');

		const scrollIntoView = vi.fn();
		card()!.scrollIntoView = scrollIntoView;
		pill.click();
		expect(scrollIntoView).toHaveBeenCalledWith(expect.objectContaining({ block: 'center' }));
		expect(document.activeElement).toBe(card());

		observer!.fire(true);
		await settle();
		expect(target.querySelector('[data-testid="action-jump"]')).toBeNull();
	});

	it('disconnects the card observer on destroy', async () => {
		await mountWithPendingCard();
		const observer = cardObserver()!;
		component!.$destroy();
		component = undefined;
		expect(observer.disconnected).toBe(true);
	});
});
