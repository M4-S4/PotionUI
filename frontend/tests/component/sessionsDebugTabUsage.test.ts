// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AdminChatCallTrace, AdminChatSessionSummary } from '$lib/services/admin-api';
import { settle } from './helpers/settle';

const mockGetAdminChatSessions = vi.fn();
const mockGetAdminChatSessionDetail = vi.fn();

vi.mock('$lib/services/admin-api', () => ({
	getAdminChatSessions: (...args: unknown[]) => mockGetAdminChatSessions(...args),
	getAdminChatSessionDetail: (...args: unknown[]) => mockGetAdminChatSessionDetail(...args),
	clearChatCallTraces: vi.fn(),
	clearAdminChatSessions: vi.fn(),
	adminBulkDeleteChatSessions: vi.fn()
}));

vi.mock('$lib/stores/confirm', () => ({
	confirmDialog: vi.fn(async () => true)
}));

const { default: SessionsDebugTab } = await import('../../src/routes/admin/components/SessionsDebugTab.svelte');
const { createClassComponent } = await import('svelte/legacy');

const SESSION: AdminChatSessionSummary = {
	id: 'sess-1',
	user_id: 'user-1',
	mode: 'chat',
	name: 'A conversation',
	status: 'active',
	llm_config_id: 'cfg-1',
	created_at: '2026-10-27T10:00:00+00:00',
	updated_at: '2026-10-27T10:05:00+00:00',
	username: 'someuser',
	email: 'someuser@example.com',
	message_count: 2
};

function trace(overrides: Partial<AdminChatCallTrace> = {}): AdminChatCallTrace {
	return {
		id: 't1',
		session_id: 'sess-1',
		user_id: 'user-1',
		message_id: 'msg-2',
		purpose: 'chat_tools',
		iteration: 1,
		provider: 'anthropic',
		model: 'claude-sonnet-5-5',
		request_system: null,
		request_messages: [{ role: 'user', content: 'hello' }],
		request_params: {},
		request_tools: null,
		response_text: 'hi',
		response_tool_calls: null,
		prompt_tokens: 5210,
		completion_tokens: 312,
		duration_ms: 1840,
		created_at: '2026-10-27T10:00:01+00:00',
		...overrides
	};
}

function message(id: string, role: 'user' | 'assistant', content: string) {
	return {
		id,
		session_id: 'sess-1',
		role,
		content,
		parsed_content: null,
		created_at: '2026-10-27T10:00:00+00:00',
		tokens_used: null,
		prompt_tokens: null,
		completion_tokens: null,
		tool_executions: null,
		metadata: null
	};
}

let destroy: (() => void) | undefined;

async function openTraces(traces: AdminChatCallTrace[]) {
	mockGetAdminChatSessions.mockResolvedValue({
		success: true,
		data: { sessions: [SESSION], total: 1, limit: 20, offset: 0, tracing_enabled: true }
	});
	mockGetAdminChatSessionDetail.mockResolvedValue({
		success: true,
		data: {
			session: {
				...SESSION,
				original_text: null,
				title_generated: false,
				closed_at: null,
				metadata: null,
				messages: [message('msg-1', 'user', 'hello'), message('msg-2', 'assistant', 'hi')]
			},
			traces
		}
	});
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({ component: SessionsDebugTab as never, target, props: {} });
	destroy = () => {
		component.$destroy();
		target.remove();
	};
	await settle();
	(target.querySelector('.dt-scroll > [role="row"]') as HTMLElement).click();
	await settle();
	return target;
}

const summary = (target: HTMLElement, index = 0) =>
	target.querySelectorAll<HTMLElement>('[data-testid="trace-summary"]')[index];

async function expand(target: HTMLElement, index = 0) {
	(summary(target, index).closest('button') as HTMLButtonElement).click();
	await settle();
}

afterEach(() => {
	destroy?.();
	destroy = undefined;
	document.body.innerHTML = '';
	vi.clearAllMocks();
});

describe('SessionsDebugTab — usage and cost on a call trace', () => {
	const usage = {
		input_tokens: 5210,
		output_tokens: 312,
		reasoning_tokens: 180,
		cached_input_tokens: 4800,
		cache_write_tokens: 0,
		cost: 0.0031
	};

	it('puts cached tokens, reasoning tokens and cost in the call header', async () => {
		const target = await openTraces([trace({ adapter: 'anthropic-messages', usage, cost: 0.0031 })]);

		const text = summary(target).textContent!.replace(/\s+/g, ' ');
		expect(text).toContain('anthropic/claude-sonnet-5-5 · anthropic-messages');
		expect(text).toContain('5,210→312 tok · 4,800 cached · 180 reasoning · $0.0031');
	});

	it('lists every figure in a usage grid once the call is opened', async () => {
		const target = await openTraces([trace({ usage, cost: 0.0031 })]);
		expect(target.querySelector('[data-testid="trace-usage"]')).toBeNull();

		await expand(target);

		const grid = target.querySelector('[data-testid="trace-usage"]')!;
		const pairs = [...grid.querySelectorAll('dt')].map((label) => [
			label.textContent?.trim(),
			label.nextElementSibling?.textContent?.trim()
		]);
		expect(pairs).toEqual([
			['Input', '5,210'],
			['Output', '312'],
			['Cached input', '4,800'],
			['Cache write', '0'],
			['Reasoning', '180'],
			['Cost', '$0.0031']
		]);
	});

	it('shows the wire request and the raw response when the trace has them', async () => {
		const target = await openTraces([
			trace({
				request_body: { model: 'claude-sonnet-5-5', max_tokens: 1000 },
				response_raw: 'event: message_start'
			})
		]);

		await expand(target);

		const text = target.textContent ?? '';
		expect(text).toContain('Wire request');
		expect(text).toContain('"max_tokens": 1000');
		expect(text).toContain('Raw response');
		expect(text).toContain('event: message_start');
	});

	it('keeps an older trace to its plain token line, with only input and output in the grid and no wire sections', async () => {
		const target = await openTraces([trace()]);

		expect(summary(target).textContent!.replace(/\s+/g, ' ')).toContain('5,210→312 tok');
		expect(summary(target).textContent).not.toContain('cached');
		expect(summary(target).textContent).not.toContain('$');
		await expand(target);

		const text = target.textContent ?? '';
		expect(text).not.toContain('Wire request');
		expect(text).not.toContain('Raw response');
		expect(target.querySelector('[data-testid="trace-usage"]')?.textContent).toContain('Input');
		expect(target.querySelector('[data-testid="trace-usage"]')?.textContent).not.toContain('Cost');
	});
});
