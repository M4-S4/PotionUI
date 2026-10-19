// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getOrganizeProvenance: vi.fn(() => Promise.resolve({ success: true, data: [] })),
		getClient: vi.fn(() => ({
			get: vi.fn().mockRejectedValue(new Error('not mocked')),
			post: vi.fn().mockRejectedValue(new Error('not mocked')),
			put: vi.fn().mockRejectedValue(new Error('not mocked'))
		})),
		getGenerationParams: vi.fn().mockResolvedValue({ success: true, data: { parameters: {}, models: [] } }),
		getGenerationById: vi.fn().mockResolvedValue({ success: false, error: 'not mocked' }),
		getTags: vi.fn().mockResolvedValue({ success: true, data: { tags: [] } }),
		getBaseURL: vi.fn(() => ''),
		getToken: vi.fn(() => null),
		setOnAuthExpired: vi.fn(),
		setOnTokenRefreshed: vi.fn()
	}
}));

const viewerLoads = vi.hoisted(() => [] as Array<{ url: string; format: string | null }>);

vi.mock('$lib/components/workbench/renderers/splat/splatViewer', () => ({
	createSplatViewer: () => ({
		load: (url: string, format: string | null) => {
			viewerLoads.push({ url, format });
			return new Promise(() => {});
		},
		clear: () => {},
		resetView: () => {},
		setAutoRotate: () => {},
		dispose: () => {}
	})
}));

const { default: GenerationDetailsModal } = await import(
	'$lib/components/modals/GenerationDetailsModal.svelte'
);
const { createClassComponent } = await import('svelte/legacy');

async function settle() {
	for (let i = 0; i < 8; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

function baseGeneration(files: Record<string, unknown>[]) {
	return {
		id: 'gen-splat-details',
		form_data: {},
		status: 'completed' as const,
		progress: 1,
		created_at: '2026-01-01T00:00:00Z',
		updated_at: '2026-01-01T00:00:00Z',
		files,
		segments: [],
		rating: 0,
		is_favorite: false
	};
}

function mountModal(generation: ReturnType<typeof baseGeneration>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: GenerationDetailsModal as never,
		target,
		props: { generation: generation as never, isOpen: true }
	});
	return {
		target,
		component,
		formatBadge: () =>
			Array.from(document.body.querySelectorAll('span.uppercase')).find((el) =>
				/^[A-Z0-9]+$/.test(el.textContent?.trim() ?? '')
			)?.textContent?.trim(),
		hasImg: () => !!document.body.querySelector('img'),
		hasVideo: () => !!document.body.querySelector('video'),
		bodyText: () => document.body.textContent ?? '',
		destroy: () => {
			component.$destroy();
			target.remove();
		}
	};
}

let mounted: ReturnType<typeof mountModal> | undefined;

beforeEach(() => {
	viewerLoads.length = 0;
});

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
});

describe('GenerationDetailsModal splat support', () => {
	it('routes an uppercase SPLAT file_type to the splat viewer, not the image fallback', async () => {
		mounted = mountModal(
			baseGeneration([
				{
					id: 1,
					file_type: 'SPLAT',
					is_final: true,
					splat_format: 'ply',
					file_path: 'generations/2026-10-19/gen-splat-details/a.ply',
					created_at: '2026-01-01T00:00:00Z'
				}
			])
		);
		await settle();

		expect(mounted.formatBadge()).toBe('PLY');
		expect(mounted.hasImg()).toBe(false);
		expect(mounted.hasVideo()).toBe(false);
		expect(viewerLoads).toEqual([{ url: '/api/media/generations/gen-splat-details/a.ply', format: 'ply' }]);
		expect(mounted.bodyText()).toContain('Loading scene');
	});

	it('takes the format from the file name when the row carries none', async () => {
		mounted = mountModal(
			baseGeneration([
				{
					id: 1,
					file_type: 'Splat',
					is_final: true,
					file_path: 'a.spz',
					created_at: '2026-01-01T00:00:00Z'
				}
			])
		);
		await settle();

		expect(mounted.formatBadge()).toBe('SPZ');
		expect(viewerLoads[0].format).toBe('spz');
	});

	it('does not offer a splat to the library, which holds images, video and audio', async () => {
		mounted = mountModal(
			baseGeneration([
				{
					id: 1,
					file_type: 'SPLAT',
					is_final: true,
					file_path: 'a.ply',
					created_at: '2026-01-01T00:00:00Z'
				}
			])
		);
		await settle();

		const labels = Array.from(document.body.querySelectorAll('button')).map(
			(button) => button.getAttribute('aria-label') ?? button.textContent ?? ''
		);
		expect(labels.some((label) => /copy to library/i.test(label))).toBe(false);
	});

	it('still renders the img fallback for a plain image file (control case)', async () => {
		mounted = mountModal(
			baseGeneration([
				{
					id: 1,
					file_type: 'image',
					is_final: true,
					file_path: 'a.png',
					created_at: '2026-01-01T00:00:00Z'
				}
			])
		);
		await settle();

		expect(mounted.formatBadge()).toBe('PNG');
		expect(mounted.hasImg()).toBe(true);
		expect(viewerLoads).toEqual([]);
	});
});
