// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { flushSync } from 'svelte';
import type { GenerationState, Tab } from '$lib/types/tabs';

const probes = vi.hoisted(() => ({ settle: [] as Array<() => void> }));

vi.mock('$lib/generation/videoHandoff', async (importOriginal) => ({
	...(await importOriginal<Record<string, unknown>>()),
	preloadVideo: (_url: string, onSettled: () => void) => probes.settle.push(onSettled)
}));

const { default: StudioCanvas } = await import('../../src/routes/generate/components/studio/StudioCanvas.svelte');
const { createClassComponent } = await import('svelte/legacy');

const PREVIEW = 'data:image/png;base64,last-preview';
const VIDEO_URL = '/api/media/generations/gen-1/0.mp4';

function generation(overrides: Partial<GenerationState> = {}): GenerationState {
	return {
		isGenerating: false,
		currentGeneration: null,
		currentProgress: null,
		pipeTimers: {},
		startedAt: null,
		totalTime: null,
		lastDurationMs: null,
		batchImages: [],
		batchVideos: [],
		batchAudios: [],
		artifacts: [],
		workbenchIndex: 0,
		workbenchTotal: 0,
		queue: [],
		submittedPromptTemplate: null,
		...overrides
	} as GenerationState;
}

const tabWith = (gen: GenerationState) => ({ id: 'deck-tab', name: 'Tab', generation: gen }) as unknown as Tab;

let target: HTMLElement | undefined;
let component: { $destroy: () => void; $set: (props: Record<string, unknown>) => void } | undefined;

afterEach(() => {
	component?.$destroy();
	component = undefined;
	target?.remove();
	target = undefined;
	probes.settle = [];
});

describe('studio canvas video result landing', () => {
	it('keeps the last preview until the final video has loaded', () => {
		target = document.createElement('div');
		document.body.appendChild(target);
		component = createClassComponent({
			component: StudioCanvas as never,
			target,
			props: {
				tab: tabWith(
					generation({
						isGenerating: true,
						currentGeneration: { id: 'gen-1', generation_id: 'gen-1', status: 'running', file_type: 'video', current_image: PREVIEW } as never
					})
				)
			}
		}) as unknown as typeof component;
		flushSync();
		expect(target.querySelector(`img[src="${PREVIEW}"]`)).not.toBeNull();

		component!.$set({
			tab: tabWith(
				generation({
					isGenerating: false,
					currentGeneration: { id: 'gen-1', generation_id: 'gen-1', status: 'completed', file_type: 'video', current_image: null } as never,
					batchVideos: [{ url: VIDEO_URL, originalUrl: VIDEO_URL, file_type: 'video' }] as never,
					workbenchTotal: 1
				})
			)
		});
		flushSync();

		expect(target.querySelector(`img[src="${PREVIEW}"]`)).not.toBeNull();
		expect(target.querySelector('video')).toBeNull();
		expect(probes.settle).toHaveLength(1);

		probes.settle[0]();
		flushSync();

		expect(target.querySelector('video')).not.toBeNull();
		expect(target.querySelector(`img[src="${PREVIEW}"]`)).toBeNull();
	});
});
