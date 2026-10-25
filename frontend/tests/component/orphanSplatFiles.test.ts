// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';

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
		getGenerationThumbnailURL: vi.fn(() => '/thumb.png'),
		getBaseURL: vi.fn(() => ''),
		getToken: vi.fn(() => null),
		setOnAuthExpired: vi.fn(),
		setOnTokenRefreshed: vi.fn()
	}
}));

const { default: GenerationDetailsModal } = await import(
	'$lib/components/modals/GenerationDetailsModal.svelte'
);
const { default: GenerationCard } = await import('$lib/components/GenerationCard.svelte');
const { createClassComponent } = await import('svelte/legacy');
const { mapGenerationFiles } = await import('$lib/utils/generationOrchestrator');
const { collectCopyableFileIds } = await import('$lib/library/copyToLibrary');

const SPLAT_ROW = {
	id: 7,
	file_type: 'SPLAT',
	is_final: true,
	is_derived: false,
	file_path: 'generations/2026-10-22/gen-splat/1.ply',
	file_size: 47_328_501,
	mime_type: 'model/x-ply',
	thumbnail_small: 'thumbnails/1_small.webp',
	thumbnail_medium: 'thumbnails/1_medium.webp',
	thumbnail_large: 'thumbnails/1_large.webp',
	created_at: '2026-10-22T10:00:00Z'
};

const IMAGE_ROW = {
	id: 8,
	file_type: 'IMAGE',
	is_final: true,
	file_path: 'generations/2026-10-22/gen-splat/2.png',
	created_at: '2026-10-22T10:00:00Z'
};

function generation(files: Record<string, unknown>[]) {
	return {
		id: `gen-${Math.random()}`,
		form_data: {},
		status: 'completed' as const,
		progress: 1,
		created_at: '2026-10-22T10:00:00Z',
		updated_at: '2026-10-22T10:00:00Z',
		files,
		segments: [],
		rating: 0,
		is_favorite: false
	};
}

const cleanups: Array<() => void> = [];

afterEach(() => {
	while (cleanups.length) cleanups.pop()?.();
});

function mount(component: unknown, props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = createClassComponent({ component: component as never, target, props });
	cleanups.push(() => {
		instance.$destroy();
		target.remove();
	});
	return target;
}

function mediaSources(root: ParentNode) {
	return Array.from(root.querySelectorAll('img, video')).map((el) => el.getAttribute('src') ?? '');
}

describe('a SPLAT file left by the removed image to splat feature', () => {
	it('is not drawn as an image in the details modal', () => {
		mount(GenerationDetailsModal, { generation: generation([SPLAT_ROW]), isOpen: true });

		expect(mediaSources(document.body).some((src) => src.endsWith('.ply'))).toBe(false);
		expect(document.body.querySelector('img, video')).toBeNull();
	});

	it('leaves the other files of the generation in the details modal', () => {
		mount(GenerationDetailsModal, { generation: generation([SPLAT_ROW, IMAGE_ROW]), isOpen: true });

		const sources = mediaSources(document.body);
		expect(sources.some((src) => src.endsWith('/2.png'))).toBe(true);
		expect(sources.some((src) => src.endsWith('.ply'))).toBe(false);
	});

	it('gives a history card the no media placeholder', () => {
		const target = mount(GenerationCard, { generation: generation([SPLAT_ROW]) });

		expect(target.textContent).toContain('No media');
		expect(target.querySelector('img, video')).toBeNull();
	});

	it('is left out when a generation is restored into the workbench', () => {
		const restored = mapGenerationFiles([SPLAT_ROW, IMAGE_ROW], 'gen-splat');

		expect(restored.totalItems).toBe(1);
		expect(restored.images).toHaveLength(1);
		expect(restored.meshes).toHaveLength(0);
	});

	it('is never sent to the library', () => {
		expect(collectCopyableFileIds([{ files: [SPLAT_ROW, IMAGE_ROW] }])).toEqual(['8']);
	});
});
