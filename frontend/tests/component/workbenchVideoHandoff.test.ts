// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { settle } from './helpers/settle';

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

const generating = {
	id: 'gen-1',
	generation_id: 'gen-1',
	status: 'running',
	file_type: 'video',
	current_image: PREVIEW
};
const completed = { id: 'gen-1', generation_id: 'gen-1', status: 'completed', file_type: 'video', current_image: null };
const previewImg = () => target.querySelector(`img[src="${PREVIEW}"]`);
const video = () => target.querySelector('[data-video-stage] video');
const stage = () => target.querySelector<HTMLElement>('[data-video-stage]');

function mountGenerating() {
	component = createClassComponent({
		component: Workbench as never,
		target,
		props: {
			emptyLayout: 'pane',
			currentGeneration: generating,
			isGenerating: true,
			workbenchIndex: 0,
			workbenchTotal: 0,
			batchImages: [],
			batchVideos: [],
			batchAudios: [],
			batchMeshes: []
		}
	}) as unknown as typeof component;
}

describe('Workbench video result landing', () => {
	it('keeps the preview until the visible player has a frame, in the order the live log shows', async () => {
		mountGenerating();
		await settle(10);
		expect(previewImg()).not.toBeNull();

		component!.$set({ currentGeneration: { ...completed, current_image: PREVIEW }, isGenerating: false });
		await settle(10);
		expect(previewImg()).not.toBeNull();
		expect(video()).toBeNull();

		component!.$set({
			currentGeneration: completed,
			workbenchTotal: 1,
			batchVideos: [{ url: VIDEO_URL, originalUrl: VIDEO_URL, file_type: 'video' }]
		});
		await settle(10);

		expect(previewImg()).not.toBeNull();
		expect(video()).not.toBeNull();
		expect(stage()!.className).toContain('opacity-0');
		const player = video();

		video()!.dispatchEvent(new Event('loadeddata'));
		await settle(10);

		expect(previewImg()).toBeNull();
		expect(stage()!.className).not.toContain('opacity-0');
		expect(video()).toBe(player);
	});

	it('swaps to the player when it errors instead of holding the preview', async () => {
		mountGenerating();
		await settle(10);
		component!.$set({
			currentGeneration: completed,
			isGenerating: false,
			workbenchTotal: 1,
			batchVideos: [{ url: VIDEO_URL, originalUrl: VIDEO_URL, file_type: 'video' }]
		});
		await settle(10);
		expect(previewImg()).not.toBeNull();

		video()!.dispatchEvent(new Event('error'));
		await settle(10);

		expect(previewImg()).toBeNull();
	});
});
