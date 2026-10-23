// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';

class StubResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}
vi.stubGlobal('ResizeObserver', StubResizeObserver);

class StubCanvasContext {
	fillStyle: unknown;
	font = '';
	textBaseline = 'alphabetic';
	imageSmoothingQuality = 'low';
	clearRect() {}
	fillRect() {}
	drawImage() {}
	fillText() {}
	setTransform() {}
}
let releaseBitmap: (() => void) | null = null;
vi.stubGlobal(
	'createImageBitmap',
	vi.fn(
		() =>
			new Promise((resolve) => {
				releaseBitmap = () => resolve({ width: 64, height: 64, close: vi.fn() });
			})
	)
);
HTMLCanvasElement.prototype.getContext = vi.fn(() => new StubCanvasContext()) as never;

const getClient = vi.fn();
const getGenerationParams = vi.fn();

vi.mock('$lib/services/api/index', () => ({
	api: {
		getClient: (...args: unknown[]) => getClient(...args),
		getGenerationParams: (...args: unknown[]) => getGenerationParams(...args)
	}
}));

const { default: HistoryStitchModal } = await import(
	'../../src/routes/history/components/HistoryStitchModal.svelte'
);
import type { MediaToolContext } from '../../src/lib/tools/tools';
import type { GenerationHistoryItem } from '../../src/lib/types/history';

const GENERATION = { id: 'gen-1', form_data: {}, segments: [] } as unknown as GenerationHistoryItem;

const CONTEXT = {
	scope: 'history',
	generations: [GENERATION],
	generationIds: ['gen-1'],
	files: [],
	items: [
		{
			id: 'gen-1:0',
			kind: 'image',
			url: '/api/media/generations/gen-1/a.png',
			filename: 'a.png',
			width: 64,
			height: 64,
			generationId: 'gen-1',
			paramIndex: 0
		}
	],
	kinds: new Set(['image']),
	collectionId: null
} as unknown as MediaToolContext;

const COMPARE = {
	cols: 1,
	rows: 1,
	xLabels: ['A'],
	yLabels: [],
	cells: [{ itemId: 'gen-1:0', failed: false, axisValues: {}, seed: null, seconds: null }]
};

let target: HTMLDivElement;
let component: ReturnType<typeof mount> | undefined;

async function settle() {
	for (let i = 0; i < 8; i++) await Promise.resolve();
	flushSync();
}

function byTestId(id: string) {
	return document.querySelector(`[data-testid="${id}"]`);
}

beforeEach(() => {
	vi.useFakeTimers();
	releaseBitmap = null;
	getClient.mockReturnValue({
		get: vi.fn().mockResolvedValue({ data: new Blob(['img'], { type: 'image/png' }) })
	});
	getGenerationParams.mockResolvedValue({ success: false });
	target = document.createElement('div');
	document.body.appendChild(target);
	component = mount(HistoryStitchModal, {
		target,
		props: { context: CONTEXT, compare: COMPARE as never, onClose: vi.fn(), onDone: vi.fn() }
	});
	flushSync();
});

afterEach(() => {
	if (component) unmount(component);
	component = undefined;
	target?.remove();
	document.body.innerHTML = '';
	vi.clearAllMocks();
	vi.useRealTimers();
});

describe('HistoryStitchModal loading layout', () => {
	it('reserves the zoom bar and size line while images load', async () => {
		await settle();

		expect(byTestId('stitch-loading')).toBeTruthy();
		expect(document.querySelector('canvas')).toBeNull();
		expect(byTestId('stitch-zoom-controls')?.classList.contains('invisible')).toBe(true);
		expect(byTestId('stitch-dimensions')?.classList.contains('invisible')).toBe(true);
	});

	it('keeps the spinner over a hidden canvas until the first draw, then reveals both together', async () => {
		await settle();
		releaseBitmap?.();
		await settle();

		const canvas = document.querySelector('canvas');
		expect(canvas).toBeTruthy();
		expect(canvas!.classList.contains('invisible')).toBe(true);
		expect(byTestId('stitch-loading')).toBeTruthy();
		expect(byTestId('stitch-dimensions')?.classList.contains('invisible')).toBe(false);

		await vi.advanceTimersByTimeAsync(150);
		flushSync();

		expect(byTestId('stitch-loading')).toBeNull();
		expect(document.querySelector('canvas')!.classList.contains('invisible')).toBe(false);
		expect(byTestId('stitch-zoom-controls')?.classList.contains('invisible')).toBe(false);
	});
});
