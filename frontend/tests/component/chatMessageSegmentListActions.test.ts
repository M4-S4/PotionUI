// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';

const { default: ChatMessage } = await import('../../src/lib/components/ChatMessage.svelte');
const { appliedSegmentActions } = await import('../../src/lib/stores/appliedSegmentActions');
const { removedSegmentActions } = await import('../../src/lib/stores/removedSegmentActions');
const { createClassComponent } = await import('svelte/legacy');

function mount(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	createClassComponent({ component: ChatMessage as never, target, props });
	return target;
}

afterEach(() => {
	document.body.innerHTML = '';
	appliedSegmentActions.clear();
	removedSegmentActions.clear('m1', 0);
});

const addContent =
	'<tool_action type="add_segment" segment_index="0" segment_id="seg-1">soft rim light</tool_action>';
const deleteContent =
	'<tool_action type="delete_segment" segment_index="1" segment_id="seg-2"></tool_action>';

function applyButton(target: HTMLElement) {
	return target.querySelector('[data-testid="segment-action-apply"]') as HTMLButtonElement;
}

describe('ChatMessage add and delete segment cards', () => {
	it('shows an add card with an Add button that reports the parsed action', () => {
		const onApplyAction = vi.fn();
		const target = mount({ role: 'assistant', content: addContent, messageId: 'm1', onApplyAction });

		expect(target.querySelector('.prompt-card-title strong')?.textContent).toBe('Add Segment');
		expect(target.querySelector('.prompt-copy')?.textContent).toContain('soft rim light');
		expect(applyButton(target).textContent?.trim()).toBe('Add');

		applyButton(target).click();
		expect(onApplyAction).toHaveBeenCalledWith(
			expect.objectContaining({ type: 'add_segment', segmentId: 'seg-1', content: 'soft rim light' }),
			0
		);
	});

	it('locks an applied add card so it cannot add twice', () => {
		appliedSegmentActions.set('new-seg', 'm1', 0);
		const target = mount({ role: 'assistant', content: addContent, messageId: 'm1', onApplyAction: () => {} });

		expect(applyButton(target).textContent?.trim()).toBe('✓ Added');
		expect(applyButton(target).disabled).toBe(true);
	});

	it('shows the text a delete card would remove and no feedback thumbs', () => {
		const target = mount({
			role: 'assistant',
			content: deleteContent,
			messageId: 'm1',
			sessionId: 's1',
			onApplyAction: () => {},
			onPromptFeedback: () => {},
			segmentTextFor: () => 'harsh shadows'
		});

		expect(target.querySelector('.prompt-card-title strong')?.textContent).toBe('Delete Segment #2');
		expect(target.querySelector('[data-testid="segment-removed-text"]')?.textContent?.trim()).toBe('harsh shadows');
		expect(applyButton(target).textContent?.trim()).toBe('Delete');
		expect(target.querySelector('.message-action.active, .prompt-card-head .message-action')).toBeNull();
		expect(target.querySelector('[data-testid="segment-action-undo"]')).toBeNull();
	});

	it('offers Undo after a delete and keeps showing the removed text', () => {
		const onUndoAction = vi.fn();
		appliedSegmentActions.set('seg-2', 'm1', 0);
		removedSegmentActions.set('m1', 0, {
			list: 'positive',
			index: 1,
			segment: { id: 'seg-2', content: 'harsh shadows' },
			previousId: 'seg-1',
			nextId: null
		});
		const target = mount({
			role: 'assistant',
			content: deleteContent,
			messageId: 'm1',
			onApplyAction: () => {},
			onUndoAction,
			segmentTextFor: () => null
		});

		expect(applyButton(target).textContent?.trim()).toBe('✓ Deleted');
		expect(target.querySelector('[data-testid="segment-removed-text"]')?.textContent?.trim()).toBe('harsh shadows');
		(target.querySelector('[data-testid="segment-action-undo"]') as HTMLButtonElement).click();
		expect(onUndoAction).toHaveBeenCalledWith(0);
	});

	it('disables add and delete where segments cannot be edited, but not update', () => {
		const updateContent =
			'<tool_action type="update_segment" segment_index="0" segment_id="seg-1">x</tool_action>';
		const target = mount({
			role: 'assistant',
			content: addContent + deleteContent + updateContent,
			messageId: 'm1',
			onApplyAction: () => {},
			listActionsDisabled: true
		});

		const buttons = Array.from(target.querySelectorAll('[data-testid="segment-action-apply"]')) as HTMLButtonElement[];
		expect(buttons.map((b) => b.disabled)).toEqual([true, true, false]);
	});
});
