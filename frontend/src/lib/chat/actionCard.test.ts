import { describe, it, expect } from 'vitest';
import type { ToolExecution, UnifiedChatMessageData } from '$lib/types/chat';
import {
	actionCardIcon,
	actionCardState,
	actionExecutions,
	actionToolLabel,
	approveVerb,
	continuationErrorText,
	collapsedSummary,
	latestPendingExecution,
	markExecutionRejected,
	MAX_REJECT_REASON_LENGTH,
	redoRequestText,
	replaceExecution
} from './actionCard';

function execution(overrides: Partial<ToolExecution> = {}): ToolExecution {
	return {
		tool_name: 'update_form_settings',
		arguments: {},
		result: { success: true, data: '' },
		duration_ms: 0,
		...overrides
	};
}

function assistant(overrides: Partial<UnifiedChatMessageData> = {}): UnifiedChatMessageData {
	return { id: 'a1', role: 'assistant', content: '', timestamp: 1, ...overrides };
}

describe('actionCardState', () => {
	it('maps every approval status to a card state', () => {
		expect(actionCardState(execution({ approval_status: 'pending' }), false)).toBe('pending');
		expect(actionCardState(execution({ approval_status: 'pending' }), true)).toBe('stale');
		expect(actionCardState(execution({ approval_status: 'approved' }), false)).toBe('applied');
		expect(actionCardState(execution({ approval_status: 'rejected' }), false)).toBe('rejected');
		expect(actionCardState(execution({ approval_status: 'failed' }), false)).toBe('failed');
	});

	it('returns null for a tool that never needed approval', () => {
		expect(actionCardState(execution({ approval_status: null }), false)).toBeNull();
		expect(actionCardState(execution(), true)).toBeNull();
	});

	it('only shows stale while the record is still pending', () => {
		expect(actionCardState(execution({ approval_status: 'approved' }), true)).toBe('applied');
	});
});

describe('actionExecutions', () => {
	it('keeps the original index of every approval-gated execution', () => {
		const items = actionExecutions([
			execution({ tool_name: 'read_form' }),
			execution({ tool_name: 'start_generation', approval_status: 'pending' }),
			execution({ tool_name: 'read_preset', skipped: true })
		]);
		expect(items.map((i) => [i.index, i.execution.tool_name])).toEqual([[1, 'start_generation']]);
	});
});

describe('latestPendingExecution', () => {
	it('finds the pending execution on the latest persisted assistant message', () => {
		const ref = latestPendingExecution([
			{ role: 'user', content: 'hi', timestamp: 0 },
			assistant({ tool_executions: [execution(), execution({ approval_status: 'pending' })] })
		]);
		expect(ref).toMatchObject({ messageId: 'a1', index: 1 });
	});

	it('skips a streaming placeholder that has no id yet', () => {
		const ref = latestPendingExecution([
			assistant({ tool_executions: [execution({ approval_status: 'pending' })] }),
			{ role: 'assistant', content: '', timestamp: 2, isStreaming: true }
		]);
		expect(ref?.messageId).toBe('a1');
	});

	it('ignores a pending execution on an older assistant message', () => {
		const ref = latestPendingExecution([
			assistant({ id: 'old', tool_executions: [execution({ approval_status: 'pending' })] }),
			{ id: 'u', role: 'user', content: 'next', timestamp: 2 },
			assistant({ id: 'new', tool_executions: [execution()] })
		]);
		expect(ref).toBeNull();
	});

	it('returns null once the execution is resolved', () => {
		expect(latestPendingExecution([assistant({ tool_executions: [execution({ approval_status: 'approved' })] })])).toBeNull();
	});
});

describe('replaceExecution / markExecutionRejected', () => {
	it('merges the new record into the indexed execution without touching others', () => {
		const messages = [
			assistant({ tool_executions: [execution({ approval_status: 'pending', preview: { action: 'x', items: [] } })] }),
			assistant({ id: 'a2', tool_executions: [execution()] })
		];
		const out = replaceExecution(messages, 'a1', 0, execution({ approval_status: 'approved' }));
		expect(out[0].tool_executions![0].approval_status).toBe('approved');
		expect(out[0].tool_executions![0].preview).toEqual({ action: 'x', items: [] });
		expect(out[1]).toBe(messages[1]);
	});

	it('rejects with the typed text as reason, capped at the contract limit', () => {
		const messages = [assistant({ tool_executions: [execution({ approval_status: 'pending' })] })];
		const out = markExecutionRejected(messages, { messageId: 'a1', index: 0 }, 'x'.repeat(MAX_REJECT_REASON_LENGTH + 50));
		expect(out[0].tool_executions![0].approval_status).toBe('rejected');
		expect(out[0].tool_executions![0].approval_reason).toHaveLength(MAX_REJECT_REASON_LENGTH);
	});

	it('is a no-op for an unknown message', () => {
		const messages = [assistant({ tool_executions: [execution({ approval_status: 'pending' })] })];
		expect(markExecutionRejected(messages, { messageId: 'nope', index: 0 }, 'r')).toBe(messages);
	});
});

describe('approveVerb', () => {
	const generation = (fields: Array<{ label: string; value: string }>) =>
		execution({ preview: { action: 'Start generation', items: [], kind: 'generation', fields } });

	it('names the image count for an image generation', () => {
		expect(approveVerb(generation([{ label: 'Mode', value: 'txt2img' }, { label: 'Batch', value: '4' }]))).toBe(
			'Run 4 images'
		);
		expect(approveVerb(generation([{ label: 'Batch', value: '1' }]))).toBe('Run 1 image');
	});

	it('falls back to a plain verb without a count or for non-image modes', () => {
		expect(approveVerb(generation([{ label: 'Mode', value: 'txt2img' }]))).toBe('Run');
		expect(approveVerb(generation([{ label: 'Mode', value: 'txt2vid' }, { label: 'Batch', value: '2' }]))).toBe('Run');
	});

	it('is Approve for everything else', () => {
		expect(approveVerb(execution())).toBe('Approve');
	});
});

describe('collapsedSummary', () => {
	it('lists the new values of a form diff', () => {
		const exec = execution({
			result: {
				success: true,
				data: JSON.stringify({
					action: 'apply_form_changes',
					applied_changes: [
						{ field_name: 'steps', old_value: 36, new_value: 24 },
						{ field_name: 'width', old_value: 1024, new_value: 1536 }
					]
				})
			}
		});
		expect(collapsedSummary(exec, 'Update Form Settings')).toBe('steps 24 · width 1536');
	});

	it('lists only the changed generation fields', () => {
		const exec = execution({
			preview: {
				action: 'Start generation',
				items: [],
				kind: 'generation',
				fields: [
					{ label: 'Mode', value: 'txt2img' },
					{ label: 'Steps', value: '24', old: '36' }
				]
			}
		});
		expect(collapsedSummary(exec, 'Run Generation')).toBe('Steps 24');
	});

	it('falls back to the preview summary', () => {
		const exec = execution({ preview: { action: 'Create category', target: 'under camera', items: [] } });
		expect(collapsedSummary(exec, 'Create Phrasebook Category')).toBe('Create category — under camera');
	});
});

describe('labels and icons', () => {
	it('prefers the catalog label and humanizes otherwise', () => {
		const catalog = [{ name: 'start_generation', description: '', hint: '', label: 'Run generation' }];
		expect(actionToolLabel('start_generation', catalog)).toBe('Run generation');
		expect(actionToolLabel('update_form_settings', catalog)).toBe('Update Form Settings');
	});

	it('picks an icon from the preview shape', () => {
		expect(actionCardIcon(execution({ preview: { action: 'a', items: [], kind: 'generation' } }))).toBe('i-sparkles');
		expect(actionCardIcon(execution({ preview: { action: 'a', items: [], kind: 'text_edit' } }))).toBe('i-edit');
		expect(
			actionCardIcon(
				execution({
					result: { success: true, data: JSON.stringify({ proposed_changes: [{ field_name: 'steps', old_value: 1, new_value: 2 }] }) }
				})
			)
		).toBe('i-sliders');
		expect(actionCardIcon(execution())).toBe('i-tools');
	});

	it('words a refused continuation from the server message, then the code', () => {
		expect(continuationErrorText({ message: 'Already continued', code: 'continuation_conflict' })).toBe('Already continued');
		expect(continuationErrorText({ code: 'continuation_conflict' })).toContain('moved on');
		expect(continuationErrorText({ code: 'tool_execution_pending' })).toContain('still waiting');
		expect(continuationErrorText({})).toBe('The reply could not continue after this action.');
	});

	it('asks the model to redo against the current form', () => {
		expect(redoRequestText('Update Segment')).toContain('"Update Segment"');
		expect(redoRequestText('Update Segment')).toContain('current form');
	});
});
