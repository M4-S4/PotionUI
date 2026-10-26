import { describe, it, expect, vi, afterEach } from 'vitest';
import { mount, unmount, flushSync, tick } from 'svelte';
import type { ToolExecution } from '../../src/lib/types/chat';
import type { ActionDecision, ActionResolveOutcome } from '../../src/lib/chat/actionCard';
import { reactiveProps } from './stubs/reactiveProps.svelte';

const { default: ChatActionCard } = await import('../../src/lib/components/chat/ChatActionCard.svelte');
const { default: RendererStub } = await import('./stubs/ChatToolRendererStub.svelte');
const { chatToolRendererRegistry } = await import('../../src/lib/registries/chatToolRendererRegistry');

const formDiff = JSON.stringify({
	status: 'pending_approval',
	proposed_changes: [
		{ field_name: 'steps', old_value: 36, new_value: 24 },
		{ field_name: 'width', old_value: 1024, new_value: 1536 }
	]
});

function execution(overrides: Partial<ToolExecution> = {}): ToolExecution {
	return {
		tool_name: 'update_form_settings',
		arguments: { changes: [{ field_name: 'steps', value: 24 }] },
		result: { success: true, data: formDiff },
		duration_ms: 0,
		approval_status: 'pending',
		...overrides
	};
}

let target: HTMLDivElement;
let component: ReturnType<typeof mount> | null = null;

function setup(overrides: Record<string, unknown> = {}) {
	target = document.createElement('div');
	document.body.appendChild(target);
	const onResolve = vi.fn<(d: ActionDecision) => Promise<ActionResolveOutcome>>(async (d) => ({
		status: d.approved ? 'approved' : 'rejected'
	}));
	const onAskRedo = vi.fn();
	const live = reactiveProps<Record<string, unknown>>({
		execution: execution(),
		actionKey: 'm1:0',
		locked: false,
		onResolve,
		onAskRedo,
		...overrides
	});
	component = mount(ChatActionCard as any, { target, props: live.props as any });
	flushSync();
	return { live, onResolve: (overrides.onResolve as typeof onResolve) ?? onResolve, onAskRedo };
}

async function settle() {
	for (let i = 0; i < 5; i++) {
		await tick();
		await new Promise((resolve) => setTimeout(resolve, 0));
	}
	flushSync();
}

function card(): HTMLElement {
	return target.querySelector('[data-testid="action-card"]') as HTMLElement;
}

function button(name: string): HTMLButtonElement | undefined {
	return Array.from(target.querySelectorAll('button')).find((b) => (b.textContent || '').trim() === name);
}

function click(el: Element | undefined) {
	el?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
}

function keydown(el: Element, key: string) {
	el.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
}

afterEach(() => {
	if (component) unmount(component);
	component = null;
	target?.remove();
	chatToolRendererRegistry.unregister('plugin_tool');
});

describe('ChatActionCard pending', () => {
	it('shows the label, both decisions and a typed diff instead of raw JSON', () => {
		setup();
		expect(card().dataset.state).toBe('pending');
		expect(card().textContent).toContain('Needs approval');
		expect(button('Reject')).toBeTruthy();
		expect(card().querySelector('.action-btn.is-approve')?.textContent).toContain('Approve');
		const rows = Array.from(target.querySelectorAll('.action-diff tr')).map((r) =>
			Array.from(r.querySelectorAll('td'))
				.map((td) => td.textContent?.trim())
				.join(' ')
		);
		expect(rows).toEqual(['steps 36→24', 'width 1024→1536']);
		expect(target.textContent).not.toContain('proposed_changes');
		expect(target.textContent).not.toContain('{');
		expect(target.querySelector('.action-hint')?.textContent).toContain('reject it with your message');
	});

	it('names the run for a generation and shows its prompt', () => {
		setup({
			execution: execution({
				tool_name: 'start_generation',
				result: { success: true, data: '' },
				preview: {
					action: 'Start generation',
					items: [],
					kind: 'generation',
					fields: [
						{ label: 'Mode', value: 'txt2img' },
						{ label: 'Resolution', value: '1536×640', mono: true },
						{ label: 'Batch', value: '4' }
					],
					text_blocks: [{ label: 'Prompt', text: 'a lone lighthouse keeper at dusk' }]
				}
			})
		});
		expect(card().querySelector('.action-btn.is-approve')?.textContent?.trim()).toBe('Run 4 images↵');
		expect(target.textContent).toContain('a lone lighthouse keeper at dusk');
		expect(target.textContent).toContain('1536×640');
	});

	it('approves on click and on Enter while the card itself is focused', async () => {
		const { onResolve } = setup();
		click(card().querySelector('.action-btn.is-approve') ?? undefined);
		await settle();
		expect(onResolve).toHaveBeenCalledWith({ approved: true });

		onResolve.mockClear();
		card().focus();
		keydown(card(), 'Enter');
		await settle();
		expect(onResolve).toHaveBeenCalledWith({ approved: true });
	});

	it('does not approve on Enter from a nested control', async () => {
		const { onResolve } = setup();
		keydown(button('Reject')!, 'Enter');
		await settle();
		expect(onResolve).not.toHaveBeenCalled();
	});

	it('opens an inline reason row and Enter submits the rejection with the reason', async () => {
		const { onResolve } = setup();
		click(button('Reject'));
		await settle();
		const input = target.querySelector('.action-reason input') as HTMLInputElement;
		expect(input).toBeTruthy();
		expect(document.activeElement).toBe(input);
		expect((card().querySelector('.action-btn.is-approve') as HTMLButtonElement).disabled).toBe(true);
		input.value = 'Only the ones from last week';
		input.dispatchEvent(new Event('input', { bubbles: true }));
		flushSync();
		keydown(input, 'Enter');
		await settle();
		expect(onResolve).toHaveBeenCalledWith({ approved: false, reason: 'Only the ones from last week' });
	});

	it('rejects without a reason when the row is left empty, and Cancel closes it', async () => {
		const { onResolve } = setup();
		click(button('Reject'));
		await settle();
		click(button('Cancel'));
		await settle();
		expect(onResolve).not.toHaveBeenCalled();
		click(button('Reject'));
		await settle();
		const rejectButtons = Array.from(target.querySelectorAll('.action-reason .action-btn.is-reject'));
		click(rejectButtons[0]);
		await settle();
		expect(onResolve).toHaveBeenCalledWith({ approved: false });
	});

	it('disables every decision while locked', () => {
		setup({ locked: true });
		expect(button('Reject')?.disabled).toBe(true);
		expect((card().querySelector('.action-btn.is-approve') as HTMLButtonElement).disabled).toBe(true);
	});

	it('surfaces a resolve error on the card instead of swallowing it', async () => {
		setup({ onResolve: vi.fn(async () => Promise.reject(new Error('Backend offline'))) });
		click(card().querySelector('.action-btn.is-approve') ?? undefined);
		await settle();
		expect(target.querySelector('[role="alert"]')?.textContent).toContain('Backend offline');
		expect(card().dataset.state).toBe('pending');
	});
});

describe('ChatActionCard resolved states', () => {
	it('applied collapses to a summary line and Details reopens the diff', async () => {
		setup({ execution: execution({ approval_status: 'approved' }) });
		expect(card().dataset.state).toBe('applied');
		expect(card().textContent).toContain('Applied');
		expect(target.querySelector('.action-summary')?.textContent).toBe('steps 24 · width 1536');
		expect(target.querySelector('.action-diff')).toBeNull();
		expect(button('Reject')).toBeUndefined();
		click(button('Details'));
		await settle();
		expect(target.querySelector('.action-diff')).toBeTruthy();
		expect(button('Hide details')).toBeTruthy();
	});

	it('rejected collapses to the quoted reason', () => {
		setup({ execution: execution({ approval_status: 'rejected', approval_reason: 'make it blue' }) });
		expect(card().textContent).toContain('Rejected');
		expect(target.querySelector('.action-summary')?.textContent).toBe('"make it blue"');
	});

	it('rejected without a reason says so', () => {
		setup({ execution: execution({ approval_status: 'rejected' }) });
		expect(target.querySelector('.action-summary')?.textContent).toBe('Rejected without a reason.');
	});

	it('expired shows an inert card that cannot be approved or rejected', async () => {
		const { onResolve } = setup({ execution: execution({ approval_status: 'expired' }) });
		expect(card().dataset.state).toBe('expired');
		expect(card().textContent).toContain('Expired');
		expect(target.querySelector('.action-summary')?.textContent).toBe('This approval expired before it was decided.');
		expect(button('Reject')).toBeUndefined();
		expect(button('Retry')).toBeUndefined();
		expect(target.querySelector('.action-btn.is-approve')).toBeNull();
		click(button('Details'));
		await settle();
		expect(target.querySelector('.action-summary')).toBeNull();
		expect(onResolve).not.toHaveBeenCalled();
	});

	it('failed shows the real error and Retry resolves approved again', async () => {
		const { onResolve } = setup({
			execution: execution({
				approval_status: 'failed',
				result: { success: false, data: '', error: 'Backend "local GPU" is offline (connection refused).' }
			})
		});
		expect(card().textContent).toContain('Failed');
		expect(target.querySelector('[data-testid="action-error"]')?.textContent).toBe(
			'Backend "local GPU" is offline (connection refused).'
		);
		click(button('Retry'));
		await settle();
		expect(onResolve).toHaveBeenCalledWith({ approved: true });
	});

	it('updates in place when the record changes underneath it', () => {
		const { live } = setup();
		live.set({ execution: execution({ approval_status: 'rejected', approval_reason: 'replied instead' }) });
		flushSync();
		expect(card().dataset.state).toBe('rejected');
		expect(target.querySelector('.action-summary')?.textContent).toBe('"replied instead"');
	});
});

describe('ChatActionCard stale', () => {
	it('turns into the form-changed state, then forces or asks to redo', async () => {
		const onResolve = vi.fn<(d: ActionDecision) => Promise<ActionResolveOutcome>>(async () => ({ status: 'stale' }));
		const { onAskRedo } = setup({ onResolve });
		click(card().querySelector('.action-btn.is-approve') ?? undefined);
		await settle();
		expect(card().dataset.state).toBe('stale');
		expect(card().textContent).toContain('Form changed');
		expect(target.querySelector('.action-stale')?.textContent).toContain('You changed the form');

		click(button('Approve anyway'));
		await settle();
		expect(onResolve).toHaveBeenLastCalledWith({ approved: true, force: true });

		click(button('Ask to redo'));
		expect(onAskRedo).toHaveBeenCalledTimes(1);
	});
});

describe('ChatActionCard plugin renderer', () => {
	it('renders a registered tool renderer inside the card instead of the default preview', async () => {
		chatToolRendererRegistry.register('plugin_tool', { component: RendererStub });
		setup({ execution: execution({ tool_name: 'plugin_tool', result: { success: true, data: '' } }) });
		await settle();
		const rendered = target.querySelector('[data-testid="plugin-renderer"]') as HTMLElement;
		expect(rendered?.textContent).toBe('plugin_tool');
		expect(rendered.dataset.expanded).toBe('true');
		expect(card().contains(rendered)).toBe(true);
		expect(target.querySelector('[data-testid="action-preview"]')).toBeNull();
	});
});
