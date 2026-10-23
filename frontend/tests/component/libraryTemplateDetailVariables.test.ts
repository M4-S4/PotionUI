import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';
import type { Segment, SegmentTemplate } from '$lib/types/segments';
import type { VariablesMap } from '$lib/utils/variableDefs';
import { mountComponent } from './helpers/mount';
import { settle } from './helpers/settle';
import { openSegmentMenuLabels } from './helpers/segmentMenu';
import { conditionModal, typeConditionTrigger } from './helpers/typing';

const { api } = await import('$lib/services/api/index');
const { activeTab, tabsStore } = await import('$lib/stores/tabs');
const { libraryTabVariables } = await import('$lib/prompts/libraryTabVariables');
const { promptSegmentActionPins } = await import('$lib/stores/promptSegmentActionPins');
const { default: TemplateDetailView } = await import('../../src/routes/prompts/sections/TemplateDetailView.svelte');

const variables: VariablesMap = {
	mood: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['dramatic', 'calm'] },
	age: { type: 'number', mode: 'shuffle', min: 18, max: 80, step: 1, value: 34 }
};

const template: SegmentTemplate = { id: 'tpl-1', name: 'Portrait lighting', description: null, segments: [], tags: [] };

const slots = (): Segment[] => [
	{ id: 's1', content: 'portrait of ${subject}, soft light', chips: {}, name: 'Subject', enabled: true },
	{ id: 's2', content: 'shot on a {35mm|50mm} lens', chips: {}, name: 'Lens', enabled: true }
];

function setTabVariables(next: VariablesMap) {
	tabsStore.updateTab(get(activeTab).id, { variables: next });
}

function mountTemplate(overrides: Record<string, unknown> = {}) {
	const mounted = mountComponent(TemplateDetailView as never, {
		mode: 'edit',
		template,
		name: template.name,
		description: '',
		tagsText: '',
		editorSegments: slots(),
		saving: false,
		dirtyCount: 0,
		onBack: vi.fn(),
		onSave: vi.fn(),
		onDiscard: vi.fn(),
		onDelete: vi.fn(),
		onDuplicate: vi.fn(),
		onApply: vi.fn(),
		tabVariables: get(libraryTabVariables),
		...overrides
	});
	return { ...mounted, editors: () => Array.from(mounted.target.querySelectorAll<HTMLElement>('.inline-chip-editor[role="textbox"]')) };
}

beforeEach(() => {
	vi.spyOn(api, 'getClient').mockReturnValue({
		get: vi.fn().mockResolvedValue({ data: { data: {} } }),
		put: vi.fn().mockResolvedValue({ data: { success: true } })
	} as never);
	promptSegmentActionPins.reset();
	setTabVariables(variables);
});

afterEach(() => {
	vi.restoreAllMocks();
	tabsStore.updateTab(get(activeTab).id, { variables: {}, promptSegments: [], negativePromptSegments: [] });
	document.body.innerHTML = '';
});

describe('template detail variables', () => {
	it('offers the variables toolbar button and lists the tab variables in the aside', async () => {
		const { target } = mountTemplate();
		await settle();

		expect(target.querySelector('button[aria-label="Variables (2)"]')).not.toBeNull();
		const rows = Array.from(target.querySelectorAll('[data-testid="variables-list"] li')).map((row) =>
			row.textContent?.replace(/\s+/g, ' ').trim()
		);
		expect(rows).toEqual(['$mood shuffle · 2 options', '$age 18–80']);
	});

	it('opens the condition editor on << in a slot and offers the tab variables', async () => {
		const { editors } = mountTemplate();
		await settle();

		await typeConditionTrigger(editors()[1]);

		expect(conditionModal()).not.toBeNull();
		expect(conditionModal()!.textContent).toContain('$mood');
	});

	it('lists Insert a condition in the segment menu', async () => {
		const { target } = mountTemplate();
		await settle();

		expect(await openSegmentMenuLabels(target, 'More actions for positive segment 1 of 2')).toContain('Insert a condition…');
	});

	it('creates an undefined variable on the active tab from a slot chip', async () => {
		setTabVariables({});
		const { editors } = mountTemplate({ tabVariables: get(libraryTabVariables) });
		await settle();

		editors()[0]
			.querySelector<HTMLElement>('.variable-usage-container .chip-main')!
			.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
		await settle();
		Array.from(document.querySelectorAll('button'))
			.find((button) => button.textContent?.includes('Create this variable'))!
			.click();
		await settle();

		expect(get(activeTab).variables).toEqual({ subject: { type: 'text', value: '' } });
		expect(document.body.textContent).toContain('Prompt variables');
	});

	it('gives the manager usage and condition reads from the whole active tab, not only the template', async () => {
		tabsStore.updateTab(get(activeTab).id, {
			promptSegments: [
				{ id: 'p1', content: 'a ${mood} sky <<if $mood = "calm">>soft<<end>> and ${mood}', chips: {}, enabled: true }
			]
		});
		const { target } = mountTemplate();
		await settle();

		Array.from(target.querySelectorAll('button'))
			.find((button) => button.textContent?.trim() === 'Manage variables')!
			.click();
		await settle();

		const dialog = document.querySelector<HTMLElement>('[role="dialog"][aria-labelledby]')!;
		expect(dialog.textContent).toContain('2 uses');
		expect(dialog.textContent).toContain('Every usage shares one roll per generation.');
	});
});
