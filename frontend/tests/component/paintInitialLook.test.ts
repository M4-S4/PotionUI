import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { installFakeCanvas } from './helpers/fakeCanvas';
import { mountComponent } from './helpers/mount';
import { settle } from './helpers/settle';

const listFilters = vi.hoisted(() => vi.fn());

vi.mock('$lib/services/api/index', () => ({
	api: {
		listFilters: (...args: unknown[]) => listFilters(...args),
		getFilterLut: vi.fn(),
		createMineFilter: vi.fn(),
		updateMineFilter: vi.fn(),
		deleteMineFilter: vi.fn(),
		listUploads: vi.fn().mockResolvedValue({ success: true, data: { uploads: [], total: 0, limit: 100, offset: 0 } }),
		uploadMedia: vi.fn()
	}
}));

const { default: MediaEditors } = await import('$lib/media/editors/MediaEditors.svelte');
const { resetFilterCatalog } = await import('$lib/filters/catalog');

let restoreCanvas: () => void;

function look(id: string, extra: Record<string, unknown> = {}) {
	return {
		id,
		name: id.charAt(0).toUpperCase() + id.slice(1),
		group: 'Colour',
		order: 0,
		intensity: 60,
		source: 'builtin',
		owned: false,
		has_lut: false,
		steps: [],
		unavailable_ops: [],
		needs_plugin: null,
		revision: '1',
		...extra
	};
}

function serve(filters: unknown[]) {
	listFilters.mockResolvedValue({ schema: 1, filters, ops: [], groups: ['Colour'], load_errors: {} });
}

beforeEach(() => {
	restoreCanvas = installFakeCanvas();
	resetFilterCatalog();
	listFilters.mockReset();
	serve([look('ember'), look('noir')]);
});

afterEach(() => {
	restoreCanvas();
	document.body.innerHTML = '';
	vi.restoreAllMocks();
});

const draw = { width: 64, height: 64, background: 'white' as const, pen: null, scribble: false };

function open(initialTool: string) {
	return mountComponent(MediaEditors, {
		request: {
			kind: 'paint',
			source: { url: '', kind: 'image', fileName: 'Untitled drawing' },
			itemIndex: null,
			draw,
			initialTool
		},
		onClose: vi.fn(),
		onResult: vi.fn()
	});
}

const filterName = () => document.body.querySelector('[data-testid="filter-name"]')?.textContent ?? null;
const applyButton = () => document.body.querySelector<HTMLButtonElement>('[data-filter-apply] button');

async function ready() {
	await vi.waitFor(() => expect(document.body.querySelector('button[aria-label="Filters"]')).not.toBeNull(), {
		timeout: 15000
	});
	await settle(20);
}

describe('the image editor opens on a recent look', () => {
	it('selects the look it was asked for on the Filters tool', async () => {
		open('filters:noir');
		await ready();

		expect(document.body.querySelector('button[aria-label="Filters"]')!.getAttribute('aria-pressed')).toBe('true');
		expect(filterName()).toBe('Noir');
	});

	it('leaves Apply enabled and focused so one key press applies it', async () => {
		open('filters:ember');
		await ready();

		expect(applyButton()).not.toBeNull();
		expect(applyButton()!.disabled).toBe(false);
		await vi.waitFor(() => expect(document.activeElement).toBe(applyButton()));
	});

	it('selects no look when it was only asked for the Filters tool', async () => {
		open('filters');
		await ready();

		expect(filterName()).toBeNull();
		await new Promise((resolve) => setTimeout(resolve, 80));
		expect(document.activeElement).not.toBe(applyButton());
	});

	it('opens on Filters with nothing selected when the look no longer exists', async () => {
		open('filters:deleted');
		await ready();

		expect(document.body.querySelector('button[aria-label="Filters"]')!.getAttribute('aria-pressed')).toBe('true');
		expect(filterName()).toBeNull();
		expect(applyButton()!.disabled).toBe(true);
	});

	it('does not select a look that needs a plugin which is not enabled', async () => {
		serve([look('vhs', { needs_plugin: 'retro-tape' })]);
		open('filters:vhs');
		await ready();

		expect(filterName()).toBeNull();
	});

	it('keeps a plugin look id with a colon in it whole', async () => {
		serve([look('retro-tape:vhs', { source: 'plugin', plugin_id: 'retro-tape' })]);
		open('filters:retro-tape:vhs');
		await ready();

		expect(filterName()).toBe('Retro-tape:vhs');
	});
});
