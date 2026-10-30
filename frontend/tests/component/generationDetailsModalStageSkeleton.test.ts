// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { flushSync } from 'svelte';

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

const { default: GenerationDetailsModal } = await import('$lib/components/modals/GenerationDetailsModal.svelte');
const { createClassComponent } = await import('svelte/legacy');

const generation = {
	id: 'gen-stage',
	form_data: {},
	status: 'completed' as const,
	progress: 1,
	created_at: '2026-01-01T00:00:00Z',
	updated_at: '2026-01-01T00:00:00Z',
	segments: [],
	rating: 0,
	is_favorite: false,
	files: [{ file_path: '/outputs/gen-stage/0.png', file_type: 'image', width: 512, height: 512 }]
};

let destroy: (() => void) | undefined;

afterEach(() => {
	destroy?.();
	destroy = undefined;
});

function mountModal() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: GenerationDetailsModal as never,
		target,
		props: { generation: generation as never, isOpen: true }
	});
	destroy = () => {
		component.$destroy();
		target.remove();
	};
	flushSync();
}

const stageImage = () => document.body.querySelector<HTMLImageElement>('[data-details-stage] img');
const skeleton = () => document.body.querySelector('[data-stage-skeleton]');

describe('generation details stage', () => {
	it('shows the pulsing skeleton instead of black until the full image has loaded', () => {
		mountModal();

		expect(stageImage()).not.toBeNull();
		expect(skeleton()).not.toBeNull();
		expect(skeleton()!.className).toContain('animate-pulse');
		expect(skeleton()!.className).not.toContain('bg-black');

		stageImage()!.dispatchEvent(new Event('load'));
		flushSync();

		expect(skeleton()).toBeNull();
	});

	it('clears the skeleton when the image fails to load', () => {
		mountModal();

		stageImage()!.dispatchEvent(new Event('error'));
		flushSync();

		expect(skeleton()).toBeNull();
	});
});
