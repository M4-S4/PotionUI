import { describe, it, expect, vi, beforeEach, afterEach, type MockInstance } from 'vitest';
import type { Writable } from 'svelte/store';
import type { ChipData, SavedSegment, SegmentCategory } from '$lib/types/segments';
import { settle } from './helpers/settle';

type PageStore = Writable<{ url: URL }>;

vi.mock('$lib/services/api/index', async (importOriginal) => {
	const actual = await importOriginal<typeof import('$lib/services/api/index')>();
	return {
		...actual,
		api: {
			...actual.api,
			listSavedSegments: vi.fn(),
			listSegmentCategories: vi.fn()
		}

	};
});

const { api } = await import('$lib/services/api/index');
const page = (await import('$app/stores')).page as unknown as PageStore;
const { default: SegmentsWorkspace } = await import('../../src/routes/prompts/sections/SegmentsWorkspace.svelte');
const { createClassComponent } = await import('svelte/legacy');
const { activeTab, tabsStore } = await import('$lib/stores/tabs');
const { get } = await import('svelte/store');

const chipData: ChipData = {
	id: 'chip-1',
	categoryPath: 'lighting',
	valueId: 'val-1',
	label: 'Golden hour',
	value: 'golden hour lighting',
	allValues: [{ id: 'val-1', label: 'Golden hour', value: 'golden hour lighting' }],
	shuffle: false,
	autoRegen: false
};

const savedSegment: SavedSegment = {
	id: 'seg-1',
	name: 'Golden hour lighting',
	category_id: 'cat-light',
	type: 'content',
	content: '#lighting',
	chips: { 'chip-1': chipData },
	enabled: true,
	tags: [],
	color: null,
	description: null
};

const category: SegmentCategory = {
	id: 'cat-light',
	name: 'Lighting',
	description: 'Lighting fragments',
	color: '#F59E0B'
};

function mountWorkspace() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: SegmentsWorkspace as never,
		target,
		props: {}
	});
	return {
		target,
		destroy: () => {
			component.$destroy();
			target.remove();
		}
	};
}

let mounted: ReturnType<typeof mountWorkspace> | undefined;
let consoleErrorSpy: MockInstance<typeof console.error>;
let windowErrors: ErrorEvent[];
let onWindowError: (event: ErrorEvent) => void;

beforeEach(() => {
	page.update((current) => ({ ...current, url: new URL('http://localhost/prompts?section=segments&id=seg-1') }));
	vi.mocked(api.listSavedSegments).mockResolvedValue({
		success: true,
		data: { segments: [savedSegment] }
	} as never);
	vi.mocked(api.listSegmentCategories).mockResolvedValue({
		success: true,
		data: { categories: [category] }
	} as never);
	consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
	windowErrors = [];
	onWindowError = (event) => windowErrors.push(event);
	window.addEventListener('error', onWindowError);
});

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
	window.removeEventListener('error', onWindowError);
	consoleErrorSpy.mockRestore();
	vi.clearAllMocks();
});

describe('segments workspace edit view', () => {
	it('opens a segment carrying chips without a DataCloneError', async () => {
		mounted = mountWorkspace();
		await settle();

		expect(windowErrors).toHaveLength(0);
		const cloneErrorLogged = consoleErrorSpy.mock.calls.some((call) =>
			call.some((arg) => String(arg).includes('DataCloneError') || String(arg).includes('could not be cloned'))
		);
		expect(cloneErrorLogged).toBe(false);

		const nameField = mounted.target.querySelector('#segment-detail-name-field') as HTMLInputElement | null;
		expect(nameField?.value).toBe('Golden hour lighting');

		const contentEditor = mounted.target.querySelector('.inline-chip-editor[role="textbox"]');
		expect(contentEditor).toBeTruthy();

		const chip = mounted.target.querySelector('[data-chip-id="chip-1"]');
		expect(chip).toBeTruthy();
		expect(chip?.textContent).toContain('Golden hour');
	});

	it('renders composer chips and lists the active tab variables', async () => {
		tabsStore.updateTab(get(activeTab).id, {
			variables: { mood: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['calm', 'tense'] } }
		});
		mounted = mountWorkspace();
		await settle();

		expect(mounted.target.querySelector('.segment-composer .segment-content [data-chip-id="chip-1"].chip')).not.toBeNull();
		const rows = Array.from(mounted.target.querySelectorAll('[data-testid="variables-list"] li')).map((row) =>
			row.textContent?.replace(/\s+/g, ' ').trim()
		);
		expect(rows).toEqual(['$mood shuffle · 2 options']);
		tabsStore.updateTab(get(activeTab).id, { variables: {} });
	});
});
