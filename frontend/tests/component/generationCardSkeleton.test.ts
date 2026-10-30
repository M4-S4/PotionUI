import { describe, it, expect, vi, afterEach } from 'vitest';
import { flushSync } from 'svelte';

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

const generation = {
	id: 'gen-1',
	form_data: {},
	status: 'completed' as const,
	progress: 1,
	created_at: '2026-10-05T10:00:00Z',
	updated_at: '2026-10-05T10:00:00Z',
	rating: 0,
	is_favorite: false,
	files: [
		{
			id: 'file-1',
			file_path: '/outputs/gen-1/0.png',
			file_type: 'image',
			thumbnail_medium: '/api/media/thumbs/gen-1/0.jpg',
			width: 512,
			height: 512
		}
	]
};

function mountCard(props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: GenerationCard as never,
		target,
		props: { generation: generation as never, showActions: false, interactive: false, ...props }
	});
	flushSync();
	return {
		target,
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

describe('generation tile loading state', () => {
	it('shows the pulsing skeleton instead of black until the thumbnail has loaded', () => {
		mounted = mountCard({ tile: { width: 240, height: 160 } });
		const skeleton = () => mounted!.target.querySelector('[data-card-skeleton]');

		expect(skeleton()).not.toBeNull();
		expect(skeleton()!.className).toContain('animate-pulse');
		expect(skeleton()!.className).not.toContain('bg-black');

		mounted.target.querySelector('img')!.dispatchEvent(new Event('load'));
		flushSync();

		expect(skeleton()).toBeNull();
	});

	it('clears the skeleton when the thumbnail fails to load', () => {
		mounted = mountCard({ tile: { width: 240, height: 160 } });

		mounted.target.querySelector('img')!.dispatchEvent(new Event('error'));
		flushSync();

		expect(mounted.target.querySelector('[data-card-skeleton]')).toBeNull();
	});

	it('leaves the fixed-aspect card without a skeleton', () => {
		mounted = mountCard({});

		expect(mounted.target.querySelector('[data-card-skeleton]')).toBeNull();
	});
});
