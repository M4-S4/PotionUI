// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { settle } from './helpers/settle';

const startMediaToolRun = vi.fn();
const getMediaToolRun = vi.fn();
const cancelMediaToolRun = vi.fn();

vi.mock('$lib/services/api/mediaTools', () => ({
	listMediaTools: vi.fn(),
	startMediaToolRun: (...args: unknown[]) => startMediaToolRun(...args),
	getMediaToolRun: (...args: unknown[]) => getMediaToolRun(...args),
	cancelMediaToolRun: (...args: unknown[]) => cancelMediaToolRun(...args),
	previewMediaTool: vi.fn()
}));
vi.mock('$lib/services/api/index', () => ({
	api: {
		getClient: vi.fn(() => ({
			get: vi.fn().mockRejectedValue(new Error('not mocked')),
			put: vi.fn().mockRejectedValue(new Error('not mocked'))
		})),
		getGenerationThumbnailURL: vi.fn(() => '/thumb.png')
	}
}));

const { mediaToolRuns } = await import('$lib/media-tools/runs');
const { default: ToolRunBand } = await import('$lib/components/tools/ToolRunBand.svelte');
const { default: LibraryCard } = await import('../../src/routes/library/components/LibraryCard.svelte');
const { default: GenerationCard } = await import('$lib/components/GenerationCard.svelte');
const { createClassComponent } = await import('svelte/legacy');

let cleanup: (() => void) | undefined;

function track(destroy: () => void, target: HTMLElement) {
	cleanup = () => {
		destroy();
		target.remove();
	};
}

function mountBand(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = mount(ToolRunBand, {
		target,
		props: { label: 'Speed', progress: 0.42, onCancel: vi.fn(), ...props }
	});
	flushSync();
	track(() => unmount(instance), target);
	return target;
}

const ITEM = {
	id: 'item-1',
	filename: 'abc.mp4',
	original_filename: 'observatory-timelapse.mp4',
	media_type: 'video',
	url: '/api/media/uploads/abc.mp4',
	width: 1920,
	height: 1080,
	duration_seconds: 8,
	tags: []
};

function mountLibraryCard(props: Record<string, unknown> = {}) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: LibraryCard as never,
		target,
		props: { item: ITEM, tile: { width: 320, height: 160 }, ...props }
	});
	track(() => component.$destroy(), target);
	return target;
}

const GENERATION = {
	id: 'gen-1',
	preset_id: 'p',
	form_data: { prompt: 'a castle' },
	status: 'completed',
	progress: 1,
	created_at: '2026-01-01T00:00:00Z',
	updated_at: '2026-01-01T00:00:00Z',
	files: [
		{
			id: 1,
			file_type: 'video',
			is_final: true,
			file_path: 'generations/2026-01-01/gen-1/0.mp4',
			width: 1920,
			height: 1080,
			duration_seconds: 4,
			created_at: '2026-01-01T00:00:00Z'
		}
	],
	rating: 0,
	is_favorite: false
};

function mountGenerationCard(props: Record<string, unknown> = {}) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: GenerationCard as never,
		target,
		props: { generation: GENERATION, tile: { width: 320, height: 160 }, ...props }
	});
	track(() => component.$destroy(), target);
	return target;
}

async function startRun(keys: string[], progress = 0.42) {
	startMediaToolRun.mockResolvedValue({ run_id: 'run-1' });
	getMediaToolRun.mockResolvedValue({
		run_id: 'run-1',
		status: 'running',
		progress,
		label: 'Encoding',
		items: [],
		replaced: false,
		error: null
	});
	const handle = await mediaToolRuns.start({
		keys,
		toolId: 'speed',
		label: 'Speed',
		body: { steps: [{ tool: 'speed', params: {} }], inputs: [{ item_id: 'item-1' }], output: { mode: 'new' } }
	});
	await settle(3);
	flushSync();
	return handle;
}

beforeEach(() => {
	startMediaToolRun.mockReset();
	getMediaToolRun.mockReset();
	cancelMediaToolRun.mockReset();
	cancelMediaToolRun.mockResolvedValue(undefined);
});

afterEach(() => {
	cleanup?.();
	cleanup = undefined;
	mediaToolRuns.destroy();
	document.body.innerHTML = '';
});

describe('ToolRunBand', () => {
	it('shows the tool label and the percent done', () => {
		const target = mountBand({ label: 'Speed', progress: 0.42 });

		expect(target.querySelector('[data-tool-run-label]')!.textContent).toBe('Speed');
		expect(target.querySelector('[data-tool-run-percent]')!.textContent).toBe('42%');
	});

	it('is a progress bar with the percent as its value', () => {
		const target = mountBand({ progress: 0.42 });

		const bar = target.querySelector('[role="progressbar"]')!;
		expect(bar.getAttribute('aria-valuenow')).toBe('42');
		expect(bar.getAttribute('aria-valuemin')).toBe('0');
		expect(bar.getAttribute('aria-valuemax')).toBe('100');
		expect(bar.getAttribute('aria-label')).toBe('Speed progress');
	});

	it('fills the thin bar to the percent', () => {
		const target = mountBand({ progress: 0.42 });

		expect((target.querySelector('[data-tool-run-fill]') as HTMLElement).style.width).toBe('42%');
	});

	it('keeps the percent within 0 to 100', () => {
		expect(mountBand({ progress: 1.7 }).querySelector('[data-tool-run-percent]')!.textContent).toBe('100%');
		cleanup?.();
		expect(mountBand({ progress: -0.3 }).querySelector('[data-tool-run-percent]')!.textContent).toBe('0%');
	});

	it('has a cancel button named for the tool', () => {
		const onCancel = vi.fn();
		const target = mountBand({ onCancel });

		const cancel = target.querySelector<HTMLButtonElement>('button[aria-label="Cancel Speed"]')!;
		cancel.click();

		expect(onCancel).toHaveBeenCalledTimes(1);
	});

	it('does not let the cancel click reach the card underneath', () => {
		const outer = vi.fn();
		document.body.addEventListener('click', outer);
		const target = mountBand({});

		target.querySelector<HTMLButtonElement>('button[aria-label="Cancel Speed"]')!.click();

		expect(outer).not.toHaveBeenCalled();
		document.body.removeEventListener('click', outer);
	});

	it('disables cancel while a cancel is in flight', () => {
		const target = mountBand({ cancelling: true });

		expect(target.querySelector<HTMLButtonElement>('button[aria-label="Cancel Speed"]')!.disabled).toBe(true);
	});

	it('sits above the info bar by the offset it is given', () => {
		const target = mountBand({ bottom: 24 });

		expect((target.querySelector('[data-tool-run]') as HTMLElement).style.bottom).toBe('24px');
	});
});

describe('Library card progress band', () => {
	it('shows no band for an item nothing is running on', async () => {
		await startRun(['someone-else']);
		const target = mountLibraryCard();

		expect(target.querySelector('[data-tool-run]')).toBeNull();
	});

	it('shows the running tool, its percent and a cancel button for the item', async () => {
		await startRun(['item-1'], 0.42);
		const target = mountLibraryCard();
		flushSync();

		expect(target.querySelector('[data-tool-run-label]')!.textContent).toBe('Speed');
		expect(target.querySelector('[data-tool-run-percent]')!.textContent).toBe('42%');
		expect(target.querySelector('button[aria-label="Cancel Speed"]')).toBeTruthy();
	});

	it('keeps the band out of the media element so its button is not nested in the tile button', async () => {
		await startRun(['item-1']);
		const target = mountLibraryCard();
		flushSync();

		const media = target.querySelector('.media-zoom')!;
		expect(media.querySelector('[data-tool-run]')).toBeNull();
		expect(target.querySelector('[data-tool-run]')).toBeTruthy();
	});

	it('rests the band on the info bar', async () => {
		await startRun(['item-1']);
		const target = mountLibraryCard({ tile: { width: 320, height: 160 } });
		flushSync();

		const bar = target.querySelector('[data-tool-run]') as HTMLElement;
		expect(bar.style.bottom).toMatch(/^\d+px$/);
		expect(parseInt(bar.style.bottom, 10)).toBeGreaterThan(0);
	});

	it('cancels the run for that item', async () => {
		await startRun(['item-1']);
		const target = mountLibraryCard();
		flushSync();

		target.querySelector<HTMLButtonElement>('button[aria-label="Cancel Speed"]')!.click();
		await settle(2);

		expect(cancelMediaToolRun).toHaveBeenCalledWith('run-1');
	});

	it('drops the band when the run ends', async () => {
		getMediaToolRun.mockResolvedValue({
			run_id: 'run-1',
			status: 'completed',
			progress: 1,
			label: '',
			items: [],
			replaced: false,
			error: null
		});
		startMediaToolRun.mockResolvedValue({ run_id: 'run-1' });
		const target = mountLibraryCard();
		const handle = await mediaToolRuns.start({
			keys: ['item-1'],
			toolId: 'speed',
			label: 'Speed',
			body: { steps: [], inputs: [], output: { mode: 'new' } }
		});
		await handle.done;
		flushSync();

		expect(target.querySelector('[data-tool-run]')).toBeNull();
	});

	it('tags a card New only when asked to', () => {
		const plain = mountLibraryCard();
		expect(plain.querySelector('[data-new-tag]')).toBeNull();
		cleanup?.();

		const fresh = mountLibraryCard({ isNew: true });
		expect(fresh.querySelector('[data-new-tag]')!.textContent).toBe('New');
	});

	it('puts the New tag in the info bar next to the resolution', () => {
		const target = mountLibraryCard({ isNew: true });

		const tag = target.querySelector('[data-new-tag]')!;
		expect(tag.parentElement!.textContent).toContain('1920×1080');
		expect(target.querySelector('.media-zoom')!.contains(tag)).toBe(false);
	});
});

describe('History card progress band', () => {
	it('shows the band for a run keyed by the generation file', async () => {
		await startRun(['gen-1:0'], 0.6);
		const target = mountGenerationCard();
		flushSync();

		expect(target.querySelector('[data-tool-run-label]')!.textContent).toBe('Speed');
		expect(target.querySelector('[data-tool-run-percent]')!.textContent).toBe('60%');
	});

	it('shows nothing for another generation', async () => {
		await startRun(['gen-2:0']);
		const target = mountGenerationCard();
		flushSync();

		expect(target.querySelector('[data-tool-run]')).toBeNull();
	});

	it('cancels by the key the run was started under', async () => {
		await startRun(['gen-1:0']);
		const target = mountGenerationCard();
		flushSync();

		target.querySelector<HTMLButtonElement>('button[aria-label="Cancel Speed"]')!.click();
		await settle(2);

		expect(cancelMediaToolRun).toHaveBeenCalledWith('run-1');
	});

	it('keeps the band out of the media element', async () => {
		await startRun(['gen-1:0']);
		const target = mountGenerationCard();
		flushSync();

		expect(target.querySelector('.media-zoom')!.querySelector('[data-tool-run]')).toBeNull();
	});
});
