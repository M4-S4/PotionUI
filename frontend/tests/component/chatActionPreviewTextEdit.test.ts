// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import type { ToolExecution } from '../../src/lib/types/chat';

const { default: ChatActionPreview } = await import('../../src/lib/components/chat/ChatActionPreview.svelte');

let target: HTMLDivElement;
let component: ReturnType<typeof mount> | null = null;

function render(execution: ToolExecution) {
	target = document.createElement('div');
	document.body.appendChild(target);
	component = mount(ChatActionPreview as any, { target, props: { execution } });
	flushSync();
	return target;
}

afterEach(() => {
	if (component) unmount(component);
	component = null;
	target?.remove();
});

function addPrompt(fields: Array<{ label: string; value: string }>): ToolExecution {
	return {
		tool_name: 'add_prompt',
		arguments: {},
		result: { success: true, data: '{}' },
		duration_ms: 0,
		approval_status: 'pending',
		preview: {
			action: 'Add prompt',
			target: 'Moonlit shelf',
			items: [],
			kind: 'text_edit',
			summary: 'a shelf of potions under moonlight',
			fields,
			text_blocks: [{ label: 'Prompt', text: 'a shelf of potions under moonlight' }]
		} as any
	};
}

describe('text edit approval preview', () => {
	it('shows the prompt name next to the prompt text', () => {
		const root = render(addPrompt([{ label: 'Name', value: 'Moonlit shelf' }, { label: 'Usage', value: 'positive' }]));
		const text = root.textContent ?? '';
		expect(text).toContain('Name');
		expect(text).toContain('Moonlit shelf');
		expect(text).toContain('Usage');
		expect(text).toContain('a shelf of potions under moonlight');
	});

	it('still renders only the text block when there are no fields', () => {
		const root = render(addPrompt([]));
		expect(root.querySelector('.action-diff')).toBeNull();
		expect(root.textContent).toContain('a shelf of potions under moonlight');
	});
});
