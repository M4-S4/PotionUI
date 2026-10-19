// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getClient: vi.fn(() => ({
			get: vi.fn().mockRejectedValue(new Error('not mocked')),
			put: vi.fn().mockRejectedValue(new Error('not mocked'))
		})),
		getGenerationThumbnailURL: vi.fn(() => '/thumb.png')
	}
}));

const { default: GenerationCard } = await import('$lib/components/GenerationCard.svelte');
const { createClassComponent } = await import('svelte/legacy');
const { iconPaths } = await import('$lib/utils/IconLibrary');

const SPLAT_FIRST_PATH = (iconPaths.splat as string[])[0];
const CUBE_PATH = iconPaths.cube as string;
const VIDEO_PATH = iconPaths.video as string;

function baseGeneration(files: Record<string, unknown>[]) {
	return {
		id: `gen-${Math.random()}`,
		form_data: {},
		status: 'completed' as const,
		progress: 1,
		created_at: '2026-01-01T00:00:00Z',
		updated_at: '2026-01-01T00:00:00Z',
		files,
		rating: 0,
		is_favorite: false
	};
}

function mountCard(generation: ReturnType<typeof baseGeneration>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: GenerationCard as never,
		target,
		props: { generation: generation as never }
	});
	return {
		target,
		component,
		formatLabel: () => target.querySelector('.font-mono.text-2xs.uppercase')?.textContent?.trim(),
		nextButton: () => target.querySelector<HTMLButtonElement>('[aria-label="Next image"]'),
		hasThumbnailImg: () => !!target.querySelector('img'),
		markerIconPath: () =>
			target.querySelector('.absolute.top-2.z-20 svg path')?.getAttribute('d'),
		destroy: () => {
			component.$destroy();
			target.remove();
		}
	};
}

let mounted: ReturnType<typeof mountCard> | undefined;

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
});

describe('GenerationCard splat support', () => {
	it('matches file_type case-insensitively and drops non-final splat files', () => {
		mounted = mountCard(
			baseGeneration([
				{
					id: 1,
					file_type: 'SPLAT',
					is_final: true,
					splat_format: 'ply',
					file_path: 'a.ply',
					created_at: '2026-01-01T00:00:00Z'
				},
				{
					id: 2,
					file_type: 'splat',
					is_final: false,
					file_path: 'b.spz',
					created_at: '2026-01-01T00:00:00Z'
				}
			])
		);

		expect(mounted.nextButton()).toBeNull();
		expect(mounted.formatLabel()).toBe('PLY');
	});

	it('reads the format marker from the file name when the row carries no splat_format', () => {
		mounted = mountCard(
			baseGeneration([
				{
					id: 1,
					file_type: 'SPLAT',
					is_final: true,
					file_path: 'generations/2026-10-19/g/0.spz',
					created_at: '2026-01-01T00:00:00Z'
				}
			])
		);

		expect(mounted.formatLabel()).toBe('SPZ');
	});

	it('prefers the generator thumbnail to the format marker once the row has one', () => {
		mounted = mountCard(
			baseGeneration([
				{
					id: 1,
					file_type: 'SPLAT',
					is_final: true,
					file_path: 'a.ply',
					thumbnail_medium: 'thumbs/a.png',
					created_at: '2026-01-01T00:00:00Z'
				}
			])
		);

		expect(mounted.hasThumbnailImg()).toBe(true);
		expect(mounted.formatLabel()).toBeUndefined();
	});

	it('draws the splat icon, not the mesh cube, on the placeholder and the type marker', () => {
		mounted = mountCard(
			baseGeneration([
				{
					id: 1,
					file_type: 'SPLAT',
					is_final: true,
					file_path: 'a.ply',
					created_at: '2026-01-01T00:00:00Z'
				},
				{
					id: 2,
					file_type: 'splat',
					is_final: true,
					file_path: 'b.ply',
					created_at: '2026-01-01T00:00:00Z'
				}
			])
		);

		expect(mounted.markerIconPath()).toBe(SPLAT_FIRST_PATH);
		expect(mounted.markerIconPath()).not.toBe(CUBE_PATH);
		expect(mounted.target.querySelector(`svg path[d="${CUBE_PATH}"]`)).toBeNull();
		expect(mounted.nextButton()).not.toBeNull();
	});

	it('gives the splat marker icon priority over video, and the mesh cube priority over the splat', () => {
		mounted = mountCard(
			baseGeneration([
				{ id: 1, file_type: 'video', is_final: true, file_path: 'v.mp4', created_at: '2026-01-01T00:00:00Z' },
				{ id: 2, file_type: 'splat', is_final: true, file_path: 's.ply', created_at: '2026-01-01T00:00:00Z' }
			])
		);
		expect(mounted.markerIconPath()).toBe(SPLAT_FIRST_PATH);
		expect(mounted.markerIconPath()).not.toBe(VIDEO_PATH);
		mounted.destroy();

		mounted = mountCard(
			baseGeneration([
				{ id: 1, file_type: 'splat', is_final: true, file_path: 's.ply', created_at: '2026-01-01T00:00:00Z' },
				{ id: 2, file_type: 'mesh', is_final: true, file_path: 'm.glb', created_at: '2026-01-01T00:00:00Z' }
			])
		);
		expect(mounted.markerIconPath()).toBe(CUBE_PATH);
	});
});
