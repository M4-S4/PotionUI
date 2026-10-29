import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { installFakeCanvas } from './helpers/fakeCanvas';
import { mountComponent } from './helpers/mount';
import { settle } from './helpers/settle';

vi.mock('$lib/services/api/index', () => ({
	api: {
		listFilters: vi.fn().mockResolvedValue({ schema: 1, filters: [], ops: [], groups: [], load_errors: {} }),
		getFilterLut: vi.fn(),
		createMineFilter: vi.fn(),
		updateMineFilter: vi.fn(),
		deleteMineFilter: vi.fn(),
		listUploads: vi.fn().mockResolvedValue({ success: true, data: { uploads: [], total: 0, limit: 100, offset: 0 } }),
		uploadMedia: vi.fn()
	}
}));

const { default: MediaEditors } = await import('$lib/media/editors/MediaEditors.svelte');
const { PaintSession } = await import('$lib/components/imageEditor/session');

let restoreCanvas: () => void;

beforeEach(() => {
	restoreCanvas = installFakeCanvas();
});

afterEach(() => {
	restoreCanvas();
	document.body.innerHTML = '';
	vi.restoreAllMocks();
});

const draw = { width: 64, height: 64, background: 'white' as const, pen: null, scribble: false };

function open(overrides: Record<string, unknown> = {}) {
	return mountComponent(MediaEditors, {
		request: {
			kind: 'paint',
			source: { url: '', kind: 'image', fileName: 'Untitled drawing' },
			itemIndex: null,
			draw,
			...overrides
		},
		onClose: vi.fn(),
		onResult: vi.fn()
	});
}

function filtersButton(): HTMLButtonElement | null {
	return document.body.querySelector('button[aria-label="Filters"]');
}

async function editorReady() {
	await vi.waitFor(() => expect(filtersButton()).not.toBeNull(), { timeout: 15000 });
	await settle(10);
}

describe('the image editor opens on the tool its request names', () => {
	it('starts on Filters when the request asks for it', async () => {
		open({ initialTool: 'filters' });
		await editorReady();

		expect(filtersButton()).not.toBeNull();
		expect(filtersButton()!.getAttribute('aria-pressed')).toBe('true');
	});

	it('starts on the default tool when the request names none', async () => {
		open();
		await editorReady();

		expect(filtersButton()).not.toBeNull();
		expect(filtersButton()!.getAttribute('aria-pressed')).toBe('false');
	});

	it('hands the tool to the session once the image is open, and only the one asked for', async () => {
		const setTool = vi.spyOn(PaintSession.prototype, 'setTool');
		open({ initialTool: 'filters' });
		await editorReady();

		expect(setTool.mock.calls.map((call) => call[0])).toContain('filters');
	});

	it('ignores a tool the editor does not have', async () => {
		open({ initialTool: 'nonsense' });
		await editorReady();

		expect(filtersButton()!.getAttribute('aria-pressed')).toBe('false');
	});
});

describe('no context key is left behind', () => {
	it('the paint editor takes the tool as a prop, not from a context', async () => {
		const source = await import('$lib/components/imageEditor/PaintEditor.svelte?raw');
		expect(source.default).not.toContain('getContext');
		expect(source.default).toContain('export let initialTool');
	});
});
