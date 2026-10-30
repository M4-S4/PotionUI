// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { settle } from './helpers/settle';

const probes = vi.hoisted(() => ({ settle: [] as Array<() => void> }));

vi.mock('$lib/services/api/index', () => ({
	api: {
		getGenerationById: vi.fn(),
		getGenerationParams: vi.fn(),
		updateGenerationTags: vi.fn(),
		getBaseURL: () => 'http://localhost',
		getToken: () => null,
		setOnAuthExpired: vi.fn(),
		setOnTokenRefreshed: vi.fn(),
		getTags: vi.fn().mockResolvedValue({ success: true, data: { tags: [] } }),
		getClient: () => ({ get: vi.fn(), post: vi.fn() })
	}
}));

vi.mock('$app/stores', () => {
	const { writable } = require('svelte/store');
	return { page: writable({ url: new URL('http://localhost/generate') }) };
});

vi.mock('$lib/generation/videoHandoff', async (importOriginal) => ({
	...(await importOriginal<Record<string, unknown>>()),
	preloadVideo: (_url: string, onSettled: () => void) => probes.settle.push(onSettled)
}));

class FakeResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}

const { default: Workbench } = await import('$lib/components/Workbench.svelte');
const { createClassComponent } = await import('svelte/legacy');

const PREVIEW = 'data:image/png;base64,last-preview';
const VIDEO_URL = '/api/media/generations/gen-1/0.mp4';

let originalRO: unknown;
let target: HTMLElement;
let component: { $destroy: () => void; $set: (props: Record<string, unknown>) => void } | undefined;

beforeEach(() => {
	probes.settle = [];
	originalRO = (globalThis as any).ResizeObserver;
	(globalThis as any).ResizeObserver = FakeResizeObserver;
	target = document.createElement('div');
	document.body.appendChild(target);
});

afterEach(() => {
	component?.$destroy();
	component = undefined;
	target.remove();
	(globalThis as any).ResizeObserver = originalRO;
	vi.clearAllMocks();
});

function mountGenerating() {
	component = createClassComponent({
		component: Workbench as never,
		target,
		props: {
			emptyLayout: 'pane',
			currentGeneration: { id: 'gen-1', generation_id: 'gen-1', status: 'running', file_type: 'video', current_image: PREVIEW },
			isGenerating: true,
			workbenchIndex: 0,
			workbenchTotal: 0,
			batchImages: [],
			batchVideos: [],
			batchAudios: [],
			batchMeshes: []
		}
	}) as never;
}

function land() {
	component!.$set({
		currentGeneration: { id: 'gen-1', generation_id: 'gen-1', status: 'completed', file_type: 'video', current_image: null },
		isGenerating: false,
		workbenchTotal: 1,
		batchVideos: [{ url: VIDEO_URL, originalUrl: VIDEO_URL, file_type: 'video' }]
	});
}

describe('Workbench video result landing', () => {
	it('keeps the last preview and no empty player until the final video has loaded', async () => {
		mountGenerating();
		await settle(10);
		expect(target.querySelector(`img[src="${PREVIEW}"]`)).not.toBeNull();

		land();
		await settle(10);

		expect(target.querySelector(`img[src="${PREVIEW}"]`)).not.toBeNull();
		expect(target.querySelector('video')).toBeNull();
		expect(probes.settle).toHaveLength(1);

		probes.settle[0]();
		await settle(10);

		expect(target.querySelector('video')).not.toBeNull();
		expect(target.querySelector(`img[src="${PREVIEW}"]`)).toBeNull();
	});
});
