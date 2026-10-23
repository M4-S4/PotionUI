import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { VariablesMap } from '$lib/utils/variableDefs';
import { mountComponent } from './helpers/mount';
import { openSegmentMenuLabels } from './helpers/segmentMenu';
import { settle } from './helpers/settle';
import { conditionModal, typeConditionTrigger } from './helpers/typing';

const { api } = await import('$lib/services/api/index');
const { promptSegmentActionPins } = await import('$lib/stores/promptSegmentActionPins');
const { default: PromptSegment } = await import('$lib/components/PromptSegment.svelte');

const variables: VariablesMap = {
	mood: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['calm', 'tense'] }
};

const MENU_LABEL = 'More actions for positive segment 1 of 1';

function mountSegment(props: Record<string, unknown> = {}) {
	const mounted = mountComponent(PromptSegment as never, {
		segment: { id: 'seg-1', content: 'portrait, soft light', type: 'content', chips: {}, enabled: true },
		index: 0,
		total: 1,
		...props
	});
	return { ...mounted, editor: () => mounted.target.querySelector<HTMLElement>('.inline-chip-editor[role="textbox"]')! };
}

beforeEach(() => {
	vi.spyOn(api, 'getClient').mockReturnValue({
		get: vi.fn().mockResolvedValue({ data: { data: {} } }),
		put: vi.fn().mockResolvedValue({ data: { success: true } })
	} as never);
	promptSegmentActionPins.reset();
});

afterEach(() => {
	vi.restoreAllMocks();
	document.body.innerHTML = '';
});

describe('segment condition availability', () => {
	it('hides Insert a condition and keeps << literal without variables or a definition callback', async () => {
		const { target, editor } = mountSegment();
		await settle();

		const labels = await openSegmentMenuLabels(target, MENU_LABEL);
		expect(labels).not.toContain('Insert a condition…');
		expect(labels).toContain('Insert a choice group');

		await typeConditionTrigger(editor());
		expect(conditionModal()).toBeNull();
		expect(editor().textContent?.endsWith('<<')).toBe(true);
	});

	it('offers Insert a condition when variables exist', async () => {
		const { target } = mountSegment({ variables });
		await settle();

		expect(await openSegmentMenuLabels(target, MENU_LABEL)).toContain('Insert a condition…');
	});

	it('offers Insert a condition when the host can define variables even if none exist', async () => {
		const { target } = mountSegment({ onVariableDefChange: vi.fn() });
		await settle();

		expect(await openSegmentMenuLabels(target, MENU_LABEL)).toContain('Insert a condition…');
	});

	it('opens << when the host can define variables even if none exist', async () => {
		const { editor } = mountSegment({ onVariableDefChange: vi.fn() });
		await settle();

		await typeConditionTrigger(editor());

		expect(conditionModal()).not.toBeNull();
	});
});
