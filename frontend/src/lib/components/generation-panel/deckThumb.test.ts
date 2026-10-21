import { describe, it, expect } from 'vitest';
import type { GenerationState } from '$lib/types/tabs';
import { deckAmbientSource, deriveDeckThumb, formatClock, hasDeckResult, kindForCategory } from './deckThumb';

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
	};
}

const preset = { id: 'flux2', name: 'FLUX.2', cover: 'cover.webp', category: 'image' };

describe('deriveDeckThumb while running', () => {
	it('shows the live preview when the family streams one', () => {
		const view = deriveDeckThumb({
			generation: generation({ currentGeneration: { id: 'g1', status: 'running', current_image: '/api/media/preview.png' } }),
			isGenerating: true,
			preset
		});
		expect(view).toEqual({ tile: 'live', src: '/api/media/preview.png' });
	});

	it("falls back to the target type's icon without a preview or for flagged previews", () => {
		const video = { ...preset, category: 'video' };
		expect(deriveDeckThumb({ generation: generation(), isGenerating: true, preset: video })).toMatchObject({ tile: 'icon', kind: 'video', pending: true });
		for (const flag of ['preview_suppressed', 'content_flagged', 'nsfw']) {
			const flagged = generation({ currentGeneration: { id: 'g1', current_image: '/p.png', [flag]: true } });
			expect(deriveDeckThumb({ generation: flagged, isGenerating: true, preset })).toMatchObject({ tile: 'icon', kind: 'image', pending: true });
		}
	});
});

describe('deriveDeckThumb at rest', () => {
	it('shows the latest image with a count for several outputs', () => {
		const view = deriveDeckThumb({
			generation: generation({ batchImages: [{ url: '/a.png' }, { url: '/b.png' }, { url: '/c.png' }, { url: '/d.png', originalUrl: '/d-full.png' }] }),
			isGenerating: false,
			preset
		});
		expect(view).toEqual({ tile: 'media', kind: 'image', src: '/d-full.png', count: 4, duration: null });
	});

	it('shows a video frame with its duration and audio as an icon tile with its duration', () => {
		expect(deriveDeckThumb({ generation: generation({ batchVideos: [{ url: '/v.mp4', duration: 5 }] }), isGenerating: false, preset })).toEqual({
			tile: 'media',
			kind: 'video',
			src: '/v.mp4',
			count: 0,
			duration: 5
		});
		expect(deriveDeckThumb({ generation: generation({ batchAudios: [{ url: '/a.wav', duration: 94 }] }), isGenerating: false, preset })).toEqual({
			tile: 'icon',
			kind: 'audio',
			count: 0,
			duration: 94,
			pending: false
		});
	});

	it('uses the 3D icon for meshes and never shows flagged media', () => {
		expect(deriveDeckThumb({ generation: generation({ batchMeshes: [{ url: '/m.glb' }] }), isGenerating: false, preset })).toMatchObject({ tile: 'icon', kind: 'mesh' });
		const flagged = generation({ batchImages: [{ url: '/x.png', content_flagged: true }] });
		expect(deriveDeckThumb({ generation: flagged, isGenerating: false, preset })).toMatchObject({ tile: 'icon', kind: 'image', pending: false });
	});

	it('uses the type icon for flagged video and NSFW results', () => {
		const video = generation({ batchVideos: [{ url: '/v.mp4', duration: 5, content_flagged: true }] });
		expect(deriveDeckThumb({ generation: video, isGenerating: false, preset })).toEqual({ tile: 'icon', kind: 'video', count: 0, duration: 5, pending: false });
		const nsfw = generation({ batchImages: [{ url: '/n.png', nsfw: true } as never] });
		expect(deriveDeckThumb({ generation: nsfw, isGenerating: false, preset })).toEqual({ tile: 'icon', kind: 'image', count: 0, duration: null, pending: false });
	});

	it('maps plugin outputs to the text or other tile', () => {
		const text = generation({ pluginOutputs: { caption_text: { msg: {}, pluginId: 'p', asset: 'a', generationId: 'g' } } });
		const other = generation({ pluginOutputs: { depth_map: { msg: {}, pluginId: 'p', asset: 'a', generationId: 'g' } } });
		expect(deriveDeckThumb({ generation: text, isGenerating: false, preset })).toMatchObject({ tile: 'icon', kind: 'text' });
		expect(deriveDeckThumb({ generation: other, isGenerating: false, preset })).toMatchObject({ tile: 'icon', kind: 'other' });
	});

	it('shows the preset cover with no result, else the family icon', () => {
		expect(deriveDeckThumb({ generation: generation(), isGenerating: false, preset })).toEqual({ tile: 'cover', preset, cover: 'cover.webp' });
		expect(deriveDeckThumb({ generation: generation(), isGenerating: false, preset: { ...preset, cover: null } })).toMatchObject({ tile: 'family' });
		expect(deriveDeckThumb({ generation: generation(), isGenerating: false, preset: null })).toEqual({ tile: 'family', preset: null });
	});
});

describe('helpers', () => {
	it('knows which views are results', () => {
		expect(hasDeckResult({ tile: 'family', preset: null })).toBe(false);
		expect(hasDeckResult({ tile: 'icon', kind: 'video', count: 0, duration: null, pending: true })).toBe(false);
		expect(hasDeckResult({ tile: 'icon', kind: 'audio', count: 0, duration: 3, pending: false })).toBe(true);
	});

	it('formats durations as m:ss and maps categories to kinds', () => {
		expect(formatClock(5)).toBe('0:05');
		expect(formatClock(94)).toBe('1:34');
		expect(formatClock(59.6)).toBe('1:00');
		expect(kindForCategory('3d')).toBe('mesh');
		expect(kindForCategory('utility')).toBe('other');
	});
});

describe('deckAmbientSource', () => {
	const rest = (overrides: Partial<GenerationState>) => deckAmbientSource(deriveDeckThumb({ generation: generation(overrides), isGenerating: false, preset }));
	const live = (current: Record<string, unknown>) => deckAmbientSource(deriveDeckThumb({ generation: generation({ currentGeneration: { id: 'g1', ...current } }), isGenerating: true, preset }));

	it('glows from an image result or the live preview', () => {
		expect(rest({ batchImages: [{ url: '/r.png' }] })).toBe('/r.png');
		expect(live({ current_image: '/p.png' })).toBe('/p.png');
	});

	it.each([
		['a content-flagged result', () => rest({ batchImages: [{ url: '/r.png', content_flagged: true }] })],
		['an NSFW result', () => rest({ batchImages: [{ url: '/r.png', nsfw: true } as never] })],
		['a flagged live preview', () => live({ current_image: '/p.png', content_flagged: true })],
		['an NSFW live preview', () => live({ current_image: '/p.png', nsfw: true })],
		['a suppressed live preview', () => live({ current_image: '/p.png', preview_suppressed: true })],
		['a video result', () => rest({ batchVideos: [{ url: '/v.mp4', duration: 5 }] })],
		['an audio result', () => rest({ batchAudios: [{ url: '/a.wav' } as never] })],
		['a text output', () => rest({ pluginOutputs: { caption_text: {} } } as never)],
		['the preset cover before any result', () => rest({})],
		['a family tile without a cover', () => deckAmbientSource(deriveDeckThumb({ generation: generation(), isGenerating: false, preset: { ...preset, cover: null } }))]
	])('never glows from %s', (_name, source) => {
		expect(source()).toBeNull();
	});
});
