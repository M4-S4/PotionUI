import { describe, it, expect, afterEach } from 'vitest';
import type { ToolExecution } from '../../src/lib/types/chat';

const { default: ChatToolRun } = await import('../../src/lib/components/chat/ChatToolRun.svelte');
const { createClassComponent } = await import('svelte/legacy');

function execution(overrides: Partial<ToolExecution> = {}): ToolExecution {
	return {
		tool_name: 'read_active_prompt',
		arguments: {},
		result: { success: true, data: '' },
		duration_ms: 300,
		...overrides
	};
}

function mount(executions: ToolExecution[]) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	createClassComponent({ component: ChatToolRun as never, target, props: { executions } });
	return target;
}

afterEach(() => {
	document.body.innerHTML = '';
});

describe('ChatToolRun', () => {
	it('renders nothing when there are no executions', () => {
		const target = mount([]);
		expect(target.querySelector('details')).toBeNull();
	});

	it('summarizes completed executions by count and humanized name', () => {
		const target = mount([
			execution({ tool_name: 'read_active_prompt' }),
			execution({ tool_name: 'read_generation_settings' })
		]);

		const details = target.querySelector('details');
		expect(details).not.toBeNull();
		expect(details?.textContent).toContain('2 tools used');
		expect(details?.textContent).toContain('Read Active Prompt');
		expect(details?.textContent).toContain('Read Generation Settings');
	});

	it('shows a failed execution\'s result.error inline and opens by default', () => {
		const target = mount([
			execution({
				tool_name: 'update_form_settings',
				result: { success: false, data: '', error: 'Field "steps" is out of range' }
			})
		]);

		const details = target.querySelector('details') as HTMLDetailsElement;
		expect(details.open).toBe(true);
		expect(details.textContent).toContain('Field "steps" is out of range');
	});

	it('excludes approval-gated executions, which render as action cards', () => {
		const target = mount([
			execution({ tool_name: 'start_generation', approval_status: 'pending' }),
			execution({ tool_name: 'update_form_settings', approval_status: 'approved' }),
			execution({ tool_name: 'read_active_prompt' })
		]);

		const details = target.querySelector('details');
		expect(details?.textContent).toContain('1 tool used');
		expect(details?.textContent).not.toContain('Start Generation');
		expect(details?.textContent).not.toContain('Update Form Settings');
	});

	it('shows a skipped sibling call as "Not run" rather than as a failure', () => {
		const target = mount([
			execution({
				tool_name: 'read_preset',
				skipped: true,
				result: { success: false, data: '', error: 'Not run: waiting for approval of start_generation.' }
			})
		]);

		const details = target.querySelector('details') as HTMLDetailsElement;
		expect(details.classList.contains('failed')).toBe(false);
		expect(details.open).toBe(false);
		const row = target.querySelector('.tool-execution.skipped');
		expect(row?.textContent).toContain('Read Preset');
		expect(row?.textContent).toContain('Not run');
		expect(target.querySelector('.tool-execution-error')).toBeNull();
	});
});
