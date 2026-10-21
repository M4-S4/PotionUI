// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import type { ToolExecution } from '../../src/lib/types/chat';

const { default: ChatMessage } = await import('../../src/lib/components/ChatMessage.svelte');
const { default: ChatActionPreview } = await import('../../src/lib/components/chat/ChatActionPreview.svelte');
const { default: ApprovalArgTree } = await import('../../src/lib/components/chat/approval/ApprovalArgTree.svelte');
const { buildArgumentTree } = await import('../../src/lib/chat/approvalPreview');
const { appliedSegmentActions } = await import('../../src/lib/stores/appliedSegmentActions');
const { createClassComponent } = await import('svelte/legacy');

const BLOCK = '<<if $age > 65>>grey hair<<else>>auburn hair<<end>>';

let target: HTMLDivElement;
let component: ReturnType<typeof mount> | null = null;

afterEach(() => {
	if (component) unmount(component);
	component = null;
	document.body.innerHTML = '';
	appliedSegmentActions.clear();
});

function renderComponent(Component: unknown, props: Record<string, unknown>) {
	target = document.createElement('div');
	document.body.appendChild(target);
	component = mount(Component as any, { target, props });
	flushSync();
	return target;
}

function renderMessage(content: string, props: Record<string, unknown> = {}) {
	target = document.createElement('div');
	document.body.appendChild(target);
	createClassComponent({
		component: ChatMessage as never,
		target,
		props: { role: 'assistant', content, messageId: 'm1', onApplyAction: () => {}, ...props }
	});
	return target;
}

const tag = (body: string) => `<tool_action type="update_segment" segment_index="0" segment_id="seg-1">${body}</tool_action>`;
const applyButton = (root: HTMLElement) => root.querySelector('[data-testid="segment-action-apply"]') as HTMLButtonElement;

describe('update_segment version card with conditions', () => {
	it('shows the condition as a read-only chip and keeps the text around it', () => {
		const root = renderMessage(tag(`portrait, ${BLOCK}, soft light`));
		const copy = root.querySelector('.prompt-copy') as HTMLElement;
		expect(copy.querySelectorAll('.condition-chip')).toHaveLength(1);
		expect(copy.textContent).toContain('portrait,');
		expect(copy.textContent).toContain(', soft light');
		expect(copy.textContent).not.toContain('<<if');
		expect(copy.querySelector('.condition-chip button[aria-label="Remove condition"]')).toBeNull();
		expect(root.querySelector('[data-testid="condition-tag-problems"]')).toBeNull();
		expect(applyButton(root).disabled).toBe(false);
	});

	it('applies a valid condition', () => {
		const onApplyAction = vi.fn();
		const root = renderMessage(tag(BLOCK), { onApplyAction });
		applyButton(root).click();
		expect(onApplyAction).toHaveBeenCalledWith(expect.objectContaining({ content: BLOCK }), 0);
	});

	it('does not open an editor from the read-only chip', () => {
		const root = renderMessage(tag(BLOCK));
		const trigger = root.querySelector('.condition-chip button') as HTMLButtonElement;
		trigger.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
		expect(document.querySelector('[role="dialog"]')).toBeNull();
	});

	it.each([
		['unclosed', 'portrait <<if $age > 65>>grey hair', 'never reaches'],
		['stray', 'portrait <<else>> grey<<end>>', 'no open'],
		['nested', '<<if $a > 1>>x<<if $b > 1>>y<<end>><<end>>', 'Nesting is not supported'],
		['bad head', '<<if $age>65>>x<<end>>', 'spaces around every operator']
	])('disables Apply and explains a %s tag', (_name, body, fragment) => {
		const onApplyAction = vi.fn();
		const root = renderMessage(tag(body), { onApplyAction });
		const problems = root.querySelector('[data-testid="condition-tag-problems"]') as HTMLElement;
		expect(problems.textContent).toContain('Apply is off');
		expect(problems.textContent?.toLowerCase()).toContain(fragment.toLowerCase());
		expect(applyButton(root).disabled).toBe(true);
		expect(applyButton(root).className).toContain('disabled:opacity-50');
		expect(applyButton(root).className).toContain('disabled:cursor-not-allowed');
		applyButton(root).click();
		expect(onApplyAction).not.toHaveBeenCalled();
	});

	it('flags a missing variable on the chip only when the tab variables are known', () => {
		const known = renderMessage(tag(BLOCK), { variables: {} });
		expect(known.querySelector('.condition-chip.broken')).not.toBeNull();
		document.body.innerHTML = '';
		const unknown = renderMessage(tag(BLOCK));
		expect(unknown.querySelector('.condition-chip.broken')).toBeNull();
	});

	it('leaves an existing plain card alone', () => {
		const root = renderMessage(tag('soft rim light'));
		expect(root.querySelector('.condition-chip')).toBeNull();
		expect(root.querySelector('.prompt-copy')?.textContent).toContain('soft rim light');
		expect(applyButton(root).disabled).toBe(false);
	});
});

function textEdit(text: string, oldText?: string): ToolExecution {
	return {
		tool_name: 'update_saved_segment',
		arguments: {},
		result: { success: true, data: '{}' },
		duration_ms: 0,
		approval_status: 'pending',
		preview: {
			action: 'Update saved segment',
			items: [],
			kind: 'text_edit',
			fields: [],
			text_blocks: [{ label: 'Content', text, ...(oldText ? { old_text: oldText } : {}) }]
		} as any
	};
}

describe('approval preview text blocks', () => {
	it('renders conditions as chips in new and old text', () => {
		const root = renderComponent(ChatActionPreview, { execution: textEdit(`warm, ${BLOCK}`, 'warm light') });
		expect(root.querySelectorAll('.action-copy')).toHaveLength(2);
		expect(root.querySelector('.action-copy:not(.is-old) .condition-chip')).not.toBeNull();
		expect(root.querySelector('.action-copy.is-old .condition-chip')).toBeNull();
		expect(root.textContent).not.toContain('<<if');
	});

	it('flags a missing variable only when the tab variables are known', () => {
		const known = renderComponent(ChatActionPreview, { execution: textEdit(BLOCK), variables: {} });
		expect(known.querySelector('.condition-chip.broken')).not.toBeNull();
		unmount(component!);
		component = null;
		const unknown = renderComponent(ChatActionPreview, { execution: textEdit(BLOCK) });
		expect(unknown.querySelector('.condition-chip.broken')).toBeNull();
	});

	it('keeps plain text blocks as text', () => {
		const root = renderComponent(ChatActionPreview, { execution: textEdit('warm light') });
		expect(root.querySelector('.condition-chip')).toBeNull();
		expect(root.textContent).toContain('warm light');
	});
});

function variableDiff(oldValue: string, newValue: string): ToolExecution {
	return {
		tool_name: 'manage_prompt_variables',
		arguments: {},
		result: {
			success: true,
			data: JSON.stringify({
				status: 'pending_approval',
				proposed_changes: [{ field_name: '${age}', old_value: oldValue, new_value: newValue }]
			})
		},
		duration_ms: 0,
		approval_status: 'pending'
	} as ToolExecution;
}

describe('variable diff rows', () => {
	it('shows a number variable as range, step, mode and pinned value', () => {
		const root = renderComponent(ChatActionPreview, {
			execution: variableDiff('not set', 'number from 18 to 90 in steps of 1 — pinned to 70')
		});
		const value = root.querySelector('[data-testid="number-variable-value"]') as HTMLElement;
		expect(value.textContent).toContain('18–90');
		expect(value.textContent).toContain('step 1');
		expect(value.textContent).toContain('Pinned');
		expect(value.textContent).toContain('70');
	});

	it('shows the per-image and shuffle modes', () => {
		const root = renderComponent(ChatActionPreview, {
			execution: variableDiff(
				'number from 18 to 90 in steps of 1 — shuffles each generation',
				'number from 18 to 90 in steps of 1 — re-rolls independently per image'
			)
		});
		const values = [...root.querySelectorAll('[data-testid="number-variable-value"]')].map((el) => el.textContent);
		expect(values[0]).toContain('Shuffles each generation');
		expect(values[1]).toContain('Re-rolls per image');
	});

	it('keeps text variable rows as plain text', () => {
		const root = renderComponent(ChatActionPreview, { execution: variableDiff('not set', 'grey hair') });
		expect(root.querySelector('[data-testid="number-variable-value"]')).toBeNull();
		expect(root.textContent).toContain('grey hair');
	});
});

describe('argument tree fallback', () => {
	it('renders a string argument that holds a condition as a chip', () => {
		const tree = buildArgumentTree({ tool_name: 'x', arguments: { content: `a, ${BLOCK}`, note: 'plain' } } as unknown as ToolExecution);
		const root = renderComponent(ApprovalArgTree, { nodes: tree });
		expect(root.querySelectorAll('.condition-chip')).toHaveLength(1);
		expect(root.textContent).toContain('"plain"');
	});
});
