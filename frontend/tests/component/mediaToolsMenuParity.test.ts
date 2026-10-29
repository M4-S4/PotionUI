// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { A1111_TOOL, CORE_TOOLS } from './fixtures/mediaToolCatalog';
import { settle } from './helpers/settle';

vi.mock('$lib/services/api/mediaTools', () => ({
	listMediaTools: vi.fn(),
	startMediaToolRun: vi.fn(),
	getMediaToolRun: vi.fn(),
	cancelMediaToolRun: vi.fn(),
	previewMediaTool: vi.fn()
}));
vi.mock('$lib/stores/toast', () => ({
	toasts: { show: vi.fn(), success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() }
}));
vi.mock('$lib/services/api/index', () => ({
	api: {
		addUploadsToCollection: vi.fn(),
		removeUploadsFromCollection: vi.fn(),
		addToCollection: vi.fn(),
		removeFromCollection: vi.fn(),
		listCollections: vi.fn(),
		listLibraryItems: vi.fn(),
		getGenerationHistory: vi.fn(),
		getLibraryFacets: vi.fn()
	}
}));

const { api } = await import('$lib/services/api/index');
const { libraryStore } = await import('$lib/stores/library');
const { historyStore } = await import('$lib/stores/history');
const { syncCatalogTools } = await import('$lib/tools/catalogTools');
const { unregisterMediaTools } = await import('$lib/tools/tools');
const { registerCoreHistoryTools } = await import('../../src/routes/history/tools/coreTools');
const { default: LibrarySelectionToolbar } = await import(
	'../../src/routes/library/components/LibrarySelectionToolbar.svelte'
);
const { default: HistorySelectionToolbar } = await import(
	'../../src/routes/history/components/HistorySelectionToolbar.svelte'
);
const { createClassComponent } = await import('svelte/legacy');

registerCoreHistoryTools();

type Kind = 'image' | 'video' | 'audio';

interface MenuGroup {
	label: string;
	tools: string[];
	subs: Record<string, string>;
}

const EXT: Record<Kind, string> = { image: 'png', video: 'mp4', audio: 'wav' };

function libraryItem(id: string, kind: Kind) {
	return {
		id,
		filename: `${id}.${EXT[kind]}`,
		media_type: kind,
		url: `/u/${id}.${EXT[kind]}`,
		width: 640,
		height: 480,
		tags: []
	};
}

function generation(id: string, kind: Kind) {
	return {
		id,
		preset_id: 'p',
		form_data: {},
		status: 'completed',
		progress: 1,
		created_at: '2026-01-01T00:00:00Z',
		updated_at: '2026-01-01T00:00:00Z',
		files: [
			{
				id: 1,
				file_path: `out/${id}.${EXT[kind]}`,
				file_type: kind,
				is_final: true,
				width: 640,
				height: 480,
				created_at: '2026-01-01T00:00:00Z'
			}
		],
		rating: 0,
		is_favorite: false
	};
}

function mountToolbar(component: unknown) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = createClassComponent({
		component: component as never,
		target,
		props: { onBulkDeleteClick: () => {}, onToolSelect: () => {} }
	});
	return () => {
		instance.$destroy();
		target.remove();
	};
}

async function openMenu(): Promise<MenuGroup[]> {
	const trigger = Array.from(document.body.querySelectorAll('button')).find((button) =>
		button.textContent?.includes('Tools')
	)!;
	trigger.click();
	await settle();
	return Array.from(document.body.querySelectorAll('[data-tools-menu] [role="group"]')).map((group) => ({
		label: group.getAttribute('aria-label') ?? '',
		tools: Array.from(group.querySelectorAll('[data-tool]')).map((row) => row.getAttribute('data-tool') ?? ''),
		subs: Object.fromEntries(
			Array.from(group.querySelectorAll('[data-tool]')).map((row) => [
				row.getAttribute('data-tool') ?? '',
				row.querySelector('[data-tool-sub]')?.textContent ?? ''
			])
		)
	}));
}

let destroy: (() => void) | undefined;

async function libraryMenu(kind: Kind): Promise<MenuGroup[]> {
	vi.mocked(api.listLibraryItems).mockResolvedValue({
		success: true,
		data: { items: [libraryItem('u1', kind)], total: 1, limit: 24, offset: 0 }
	} as never);
	await libraryStore.load();
	libraryStore.toggleSelect('u1');
	destroy = mountToolbar(LibrarySelectionToolbar);
	await settle();
	return openMenu();
}

async function historyMenu(kind: Kind): Promise<MenuGroup[]> {
	vi.mocked(api.getGenerationHistory).mockResolvedValue({
		success: true,
		data: { generations: [generation('g1', kind)], total: 1 }
	} as never);
	await historyStore.loadGenerations();
	historyStore.toggleSelect('g1');
	destroy = mountToolbar(HistorySelectionToolbar);
	await settle();
	return openMenu();
}

const byLabel = (groups: MenuGroup[], label: string) => groups.find((group) => group.label === label);

beforeEach(() => {
	libraryStore.reset();
	historyStore.clearSelection();
	unregisterMediaTools((tool) => !!tool.catalogToolId);
	syncCatalogTools([...CORE_TOOLS, A1111_TOOL]);
	vi.mocked(api.listCollections).mockResolvedValue({ success: true, data: { collections: [], total: 0 } } as never);
	vi.mocked(api.getLibraryFacets).mockResolvedValue({ success: false } as never);
});

afterEach(() => {
	destroy?.();
	destroy = undefined;
	document.body.innerHTML = '';
	libraryStore.reset();
	historyStore.clearSelection();
	vi.clearAllMocks();
});

describe('Tools menu: Edit and Effects on both pages', () => {
	it('Library, video: Edit lists the video tools, concat disabled with its range, no Effects', async () => {
		const groups = await libraryMenu('video');

		expect(groups.map((group) => group.label).slice(0, 1)).toEqual(['Edit']);
		expect(byLabel(groups, 'Edit')!.tools).toEqual([
			'trim',
			'crop',
			'resize',
			'rotate',
			'flip',
			'speed',
			'frame',
			'concat'
		]);
		expect(byLabel(groups, 'Edit')!.subs.frame).toBe('Saves one image');
		expect(byLabel(groups, 'Edit')!.subs.concat).toBe('Select 2 to 8 items');
		expect(byLabel(groups, 'Effects')).toBeUndefined();
	});

	it('Library, image: Edit lists the image tools and Effects offers Apply a look', async () => {
		const groups = await libraryMenu('image');

		expect(byLabel(groups, 'Edit')!.tools).toEqual([
			'crop',
			'resize',
			'rotate',
			'flip',
			'fit_canvas',
			'color_key',
			'crop_subject'
		]);
		expect(byLabel(groups, 'Effects')!.tools).toEqual(['apply-look']);
		expect(groups.map((group) => group.label).slice(0, 2)).toEqual(['Edit', 'Effects']);
	});

	it('Library, audio: Edit lists trim, speed and split and nothing else', async () => {
		const groups = await libraryMenu('audio');

		expect(byLabel(groups, 'Edit')!.tools).toEqual(['trim', 'speed', 'split']);
		expect(byLabel(groups, 'Edit')!.subs.split).toBe('Saves several files');
		expect(byLabel(groups, 'Effects')).toBeUndefined();
	});

	it.each(['image', 'video', 'audio'] as const)(
		'shows History the same Edit and Effects groups as Library for %s',
		async (kind) => {
			const library = await libraryMenu(kind);
			destroy?.();
			document.body.innerHTML = '';
			libraryStore.reset();
			const history = await historyMenu(kind);

			expect(byLabel(history, 'Edit')).toEqual(byLabel(library, 'Edit'));
			expect(byLabel(history, 'Effects')).toEqual(byLabel(library, 'Effects'));
		}
	);

	it('keeps the existing groups after Edit and Effects, in the same order on both pages', async () => {
		const library = (await libraryMenu('image')).map((group) => group.label);
		destroy?.();
		document.body.innerHTML = '';
		libraryStore.reset();
		const history = (await historyMenu('image')).map((group) => group.label);

		const shared = (labels: string[]) =>
			labels.filter((label) => ['Edit', 'Effects', 'Analyze', 'Compose', 'Export'].includes(label));
		expect(shared(library)).toEqual(['Edit', 'Effects', 'Analyze', 'Compose', 'Export']);
		expect(shared(history)).toEqual(shared(library));
	});

	it('offers the A1111 export on History only, in a Plugins group with the plugin named', async () => {
		const history = await historyMenu('image');
		expect(byLabel(history, 'Plugins')!.tools).toEqual([A1111_TOOL.id]);
		expect(byLabel(history, 'Plugins')!.subs[A1111_TOOL.id]).toBe('A1111 Metadata Export');
		destroy?.();
		document.body.innerHTML = '';
		libraryStore.reset();

		const library = await libraryMenu('image');
		expect(byLabel(library, 'Plugins')).toBeUndefined();
	});
});
