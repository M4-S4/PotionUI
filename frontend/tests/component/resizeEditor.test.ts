import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const startMediaToolRun = vi.fn();
const getMediaToolRun = vi.fn();
const listUploads = vi.fn();

vi.mock('$lib/services/api/mediaTools', () => ({
	startMediaToolRun: (...args: unknown[]) => startMediaToolRun(...args),
	getMediaToolRun: (...args: unknown[]) => getMediaToolRun(...args),
	cancelMediaToolRun: vi.fn(),
	listMediaTools: vi.fn(),
	previewMediaTool: vi.fn()
}));

vi.mock('$lib/services/api/index', () => ({
	api: {
		listUploads: (...args: unknown[]) => listUploads(...args),
		listGenerationMedia: vi.fn(async () => ({ success: false })),
		copyGenerationFileToLibrary: vi.fn(async () => ({ success: false })),
		uploadMedia: vi.fn(),
		deleteLibraryItem: vi.fn(async () => ({ success: true }))
	}
}));

const { default: MediaEditors } = await import('$lib/media/editors/MediaEditors.svelte');
const { createClassComponent } = await import('svelte/legacy');
const { tick } = await import('svelte');
const { mediaToolRuns } = await import('$lib/media-tools/runs');

const clipSource = {
	url: '/api/media/uploads/clip.mp4',
	kind: 'video' as const,
	fileName: 'clip.mp4',
	itemId: 'row-2',
	storedPath: 'uploads/clip.mp4',
	width: 1920,
	height: 1080,
	durationSeconds: 8,
	fps: 24
};

const imageSource = {
	url: '/api/media/uploads/photo.png',
	kind: 'image' as const,
	fileName: 'photo.png',
	itemId: 'row-1',
	storedPath: 'uploads/photo.png',
	width: 1000,
	height: 1500
};

async function settle() {
	for (let i = 0; i < 5; i += 1) {
		await tick();
		await new Promise((resolve) => setTimeout(resolve, 0));
	}
}

async function open(source: typeof clipSource | typeof imageSource) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	createClassComponent({
		component: MediaEditors as never,
		target,
		props: { request: { kind: 'resize', source, itemIndex: null }, onClose: vi.fn(), onResult: vi.fn() }
	});
	await settle();
	return document.body;
}

function button(text: string): HTMLButtonElement {
	const found = Array.from(document.body.querySelectorAll('button')).find((candidate) =>
		(candidate.textContent || '').replace(/\s+/g, ' ').trim().startsWith(text)
	);
	if (!found) throw new Error(`No button starting with "${text}"`);
	return found;
}

function result(): string {
	return (document.body.querySelector('[data-testid="resize-result"]')?.textContent || '').trim();
}

function type(label: string, value: string) {
	const input = document.body.querySelector(`input[aria-label="${label}"]`) as HTMLInputElement;
	input.value = value;
	input.dispatchEvent(new Event('input', { bubbles: true }));
}

afterEach(() => {
	mediaToolRuns.destroy();
});

beforeEach(() => {
	document.body.innerHTML = '';
	startMediaToolRun.mockReset();
	getMediaToolRun.mockReset();
	startMediaToolRun.mockResolvedValue({ run_id: 'run-1' });
	getMediaToolRun.mockResolvedValue({
		run_id: 'run-1',
		status: 'completed',
		progress: 1,
		label: '',
		items: [{ id: 'row-2', filename: 'small.mp4', media_type: 'video', url: '/api/media/uploads/small.mp4' }],
		replaced: false,
		error: null
	});
	listUploads.mockResolvedValue({ success: true, data: { uploads: [], total: 0, limit: 100, offset: 0 } });
});

describe('resize editor', () => {
	it('offers sizes worked out from the source and preselects the largest smaller common height', async () => {
		const body = await open(clipSource);

		const text = body.textContent || '';
		for (const size of ['1920×1080', '1280×720', '854×480', '640×360', '1440×810']) {
			expect(text).toContain(size);
		}
		expect(text).not.toContain('1080p');
		expect(button('720p').getAttribute('aria-pressed')).toBe('true');
		expect(result()).toBe('1280×720');
	});

	it('sends the chosen size with the chosen quality as one resize operation', async () => {
		await open(clipSource);

		button('540p').click();
		button('Small file').click();
		await settle();
		button('Save as new').click();
		await settle();

		expect(startMediaToolRun).toHaveBeenCalledWith({
			steps: [{ tool: 'resize', params: { width: 960, height: 540, quality: 'small' } }],
			inputs: [{ item_id: 'row-2' }],
			output: { mode: 'new' }
		});
	});

	it('fills the other side of a custom size from the source shape', async () => {
		await open(clipSource);

		button('Custom size').click();
		await settle();
		type('Width', '1000');
		await settle();

		const height = document.body.querySelector('input[aria-label="Height"]') as HTMLInputElement;
		expect(height.value).toBe('562');
		expect(result()).toBe('1000×562');

		type('Height', '400');
		await settle();
		const width = document.body.querySelector('input[aria-label="Width"]') as HTMLInputElement;
		expect(width.value).toBe('712');

		button('Replace original').click();
		await settle();
		expect(startMediaToolRun).toHaveBeenCalledWith({
			steps: [{ tool: 'resize', params: { width: 712, height: 400, quality: 'balanced' } }],
			inputs: [{ item_id: 'row-2' }],
			output: { mode: 'replace' }
		});
	});

	it('refuses a custom size larger than the original', async () => {
		const body = await open(clipSource);

		button('Custom size').click();
		await settle();
		type('Width', '3840');
		await settle();

		expect(body.textContent).toContain('Larger than the original 1920×1080.');
		button('Save as new').click();
		await settle();
		expect(startMediaToolRun).not.toHaveBeenCalled();
	});

	it('refuses to save the original size', async () => {
		const body = await open(clipSource);

		button('Original').click();
		await settle();

		expect(body.textContent).toContain('Pick a size smaller than the original.');
	});

	it('resizes an image without a quality choice', async () => {
		const body = await open(imageSource);

		expect(body.textContent).not.toContain('Small file');
		button('Save as new').click();
		await settle();

		expect(startMediaToolRun).toHaveBeenCalledWith({
			steps: [{ tool: 'resize', params: { width: 720, height: 1080 } }],
			inputs: [{ item_id: 'row-1' }],
			output: { mode: 'new' }
		});
	});
});
