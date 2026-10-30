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
		getBaseURL: vi.fn(() => ''),
		getToken: vi.fn(() => null),
		setOnAuthExpired: vi.fn(),
		setOnTokenRefreshed: vi.fn()
	}
}));

vi.mock('$lib/stores/presetsCatalog', async (importOriginal) => ({
	...(await importOriginal<Record<string, unknown>>()),
	loadPresetNameMap: vi.fn(() => new Promise((resolve) => setTimeout(() => resolve({ 'krea-2/default': 'Krea-2' }), 20)))
}));

const { default: GenerationDetailsModal } = await import('$lib/components/modals/GenerationDetailsModal.svelte');
const { createClassComponent } = await import('svelte/legacy');

const generation = {
	id: 'gen-preset',
	form_data: {},
	preset_id: 'krea-2/default',
	status: 'completed' as const,
	progress: 1,
	created_at: '2026-01-01T00:00:00Z',
	updated_at: '2026-01-01T00:00:00Z',
	segments: [],
	rating: 0,
	is_favorite: false,
	files: [{ file_path: '/outputs/gen-preset/0.png', file_type: 'image', width: 512, height: 512 }]
};

let destroy: (() => void) | undefined;

afterEach(() => {
	destroy?.();
	destroy = undefined;
});

describe('GenerationDetailsModal preset name', () => {
	it('replaces Unknown preset once the preset names finish loading', async () => {
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

		expect(document.body.textContent).toContain('Unknown preset');
		await vi.waitFor(() => expect(document.body.textContent).toContain('Krea-2'));
		expect(document.body.textContent).not.toContain('Unknown preset');
	});
});
