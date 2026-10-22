import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';
import type { SavedSegment, SegmentCategory } from '$lib/types/segments';
import type { VariablesMap } from '$lib/utils/variableDefs';
import { mountComponent } from './helpers/mount';
import { settle } from './helpers/settle';
import { conditionModal, typeConditionTrigger } from './helpers/typing';

const { api } = await import('$lib/services/api/index');
const { activeTab, tabsStore } = await import('$lib/stores/tabs');
const { libraryTabVariables } = await import('$lib/prompts/libraryTabVariables');
const { default: SegmentDetailView } = await import('../../src/routes/prompts/sections/SegmentDetailView.svelte');

const variables: VariablesMap = {
	mood: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['dramatic', 'calm'] },
	age: { type: 'number', mode: 'shuffle', min: 18, max: 80, step: 1, value: 34 }
};

const category: SegmentCategory = { id: 'cat-light', name: 'Lighting', description: '', color: '#3B82F6' };

const savedSegment: SavedSegment = {
	id: 'seg-1',
	name: 'Studio portrait',
	category_id: category.id,
	type: 'content',
	content: '',
	chips: {},
	enabled: true,
	tags: [],
	color: null,
	description: null
};

function setTabVariables(next: VariablesMap) {
	tabsStore.updateTab(get(activeTab).id, { variables: next });
}

function mountSegment(content: string, overrides: Record<string, unknown> = {}) {
	const mounted = mountComponent(SegmentDetailView as never, {
		mode: 'edit',
		segment: savedSegment,
		categories: [category],
		categorySegmentCount: 1,
		name: savedSegment.name,
		categoryId: category.id,
		content,
		chips: {},
		enabled: true,
		color: '',
		description: '',
		tagsText: '',
		saving: false,
		dirtyCount: 0,
		onBack: vi.fn(),
		onSave: vi.fn(),
		onDiscard: vi.fn(),
		onDelete: vi.fn(),
		onDuplicate: vi.fn(),
		onInsert: vi.fn(),
		tabVariables: get(libraryTabVariables),
		...overrides
	});
	return { ...mounted, editor: () => mounted.target.querySelector<HTMLElement>('.inline-chip-editor[role="textbox"]')! };
}

beforeEach(() => {
	vi.spyOn(api, 'getClient').mockReturnValue({
		get: vi.fn().mockResolvedValue({ data: { data: {} } }),
		put: vi.fn().mockResolvedValue({ data: { success: true } })
	} as never);
	setTabVariables(variables);
});

afterEach(() => {
	vi.restoreAllMocks();
	tabsStore.updateTab(get(activeTab).id, { variables: {}, promptSegments: [], negativePromptSegments: [] });
	document.body.innerHTML = '';
});

describe('segment detail content editor', () => {
	it('renders the content with the composer chips inside the composer scope', async () => {
		const { target, editor } = mountSegment('portrait, {35mm|50mm} lens');
		await settle();

		expect(target.querySelector('.segment-composer .segment-content')).not.toBeNull();
		expect(editor().closest('.segment-content')).not.toBeNull();
		expect(editor().querySelector('.choice-group-container .chip')).not.toBeNull();
	});

	it('lists the active tab variables in the aside', async () => {
		const { target } = mountSegment('portrait');
		await settle();

		const rows = Array.from(target.querySelectorAll('[data-testid="variables-list"] li')).map((row) =>
			row.textContent?.replace(/\s+/g, ' ').trim()
		);
		expect(rows).toEqual(['$mood shuffle · 2 options', '$age 18–80']);
	});

	it('opens the condition editor on << and offers the tab variables', async () => {
		const { editor } = mountSegment('portrait, soft light');
		await settle();

		await typeConditionTrigger(editor());

		const dialog = conditionModal();
		expect(dialog).not.toBeNull();
		expect(dialog!.textContent).toContain('$mood');
		dialog!.querySelector<HTMLButtonElement>('[data-testid="variable-select"]')!.click();
		await settle();
		const picker = document.querySelector('[role="listbox"]');
		expect(picker?.textContent).toContain('$mood');
		expect(picker?.textContent).toContain('$age');
	});

	it('still opens the condition editor when the tab has no variables yet', async () => {
		setTabVariables({});
		const { editor } = mountSegment('portrait, soft light', { tabVariables: get(libraryTabVariables) });
		await settle();

		await typeConditionTrigger(editor());

		expect(conditionModal()).not.toBeNull();
	});

	it('creates an undefined variable on the active tab and opens the manager', async () => {
		setTabVariables({});
		const { editor, target, rerender } = mountSegment('portrait of ${subject}', { tabVariables: get(libraryTabVariables) });
		await settle();

		editor()
			.querySelector<HTMLElement>('.variable-usage-container .chip-main')!
			.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
		await settle();
		Array.from(document.querySelectorAll('button'))
			.find((button) => button.textContent?.includes('Create this variable'))!
			.click();
		await settle();

		expect(get(activeTab).variables).toEqual({ subject: { type: 'text', value: '' } });
		expect(document.body.textContent).toContain('Prompt variables');

		rerender({ tabVariables: get(libraryTabVariables) });
		await settle();
		const rows = Array.from(target.querySelectorAll('[data-testid="variables-list"] li')).map((row) =>
			row.textContent?.replace(/\s+/g, ' ').trim()
		);
		expect(rows).toEqual(['$subject text']);
	});

	it('opens the manager from the aside and writes its changes to the active tab', async () => {
		const { target } = mountSegment('portrait');
		await settle();

		Array.from(target.querySelectorAll('button'))
			.find((button) => button.textContent?.trim() === 'Manage variables')!
			.click();
		await settle();
		Array.from(document.querySelectorAll('button'))
			.find((button) => button.textContent?.trim() === 'Add variable')!
			.click();
		await settle();
		const nameField = document.querySelector<HTMLInputElement>('input[aria-label="Variable name"]')!;
		nameField.value = 'palette';
		nameField.dispatchEvent(new Event('input', { bubbles: true }));
		await settle();

		expect(Object.keys(get(activeTab).variables ?? {})).toEqual(['mood', 'age', 'palette']);
	});

	it('gives the manager usage and condition reads from the whole active tab, not only the draft', async () => {
		tabsStore.updateTab(get(activeTab).id, {
			promptSegments: [
				{ id: 'p1', content: 'a ${mood} sky <<if $mood = "calm">>soft<<end>> and ${mood}', chips: {}, enabled: true }
			]
		});
		const { target } = mountSegment('portrait');
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
