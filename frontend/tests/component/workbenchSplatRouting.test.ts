// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

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

const viewerLoads = vi.hoisted(() => [] as string[]);

vi.mock('$lib/components/workbench/renderers/splat/splatViewer', () => ({
	createSplatViewer: () => ({
		load: (url: string) => {
			viewerLoads.push(url);
			return new Promise(() => {});
		},
		clear: () => {},
		resetView: () => {},
		setAutoRotate: () => {},
		dispose: () => {}
	})
}));

vi.mock('@google/model-viewer', () => ({}));

class FakeResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}

const { default: Workbench } = await import('$lib/components/Workbench.svelte');
const { createClassComponent } = await import('svelte/legacy');

async function settle() {
	for (let i = 0; i < 12; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

let originalRO: unknown;
let target: HTMLElement;
let component: { $destroy: () => void } | undefined;

beforeEach(() => {
	originalRO = (globalThis as any).ResizeObserver;
	(globalThis as any).ResizeObserver = FakeResizeObserver;
	viewerLoads.length = 0;

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

const SPLAT_URL = '/api/media/generations/gen-1/0.ply';

function mount(props: Record<string, unknown>) {
	component = createClassComponent({
		component: Workbench as never,
		target,
		props: {
			isGenerating: false,
			workbenchIndex: 0,
			workbenchTotal: 1,
			batchImages: [],
			batchVideos: [],
			batchAudios: [],
			batchMeshes: [],
			...props
		}
	});
}

function mountWithSplat(fileType: string | undefined) {
	mount({
		currentGeneration: { status: 'completed' },
		batchSplats: [
			{
				url: SPLAT_URL,
				originalUrl: SPLAT_URL,
				...(fileType ? { file_type: fileType } : {}),
				splat_name: '0.ply',
				splat_format: 'ply'
			}
		]
	});
}

describe('Workbench routes a splat gallery item to the splat renderer', () => {
	it.each(['SPLAT', 'splat', undefined])(
		'does not render a broken <img> and loads the file in the viewer for file_type %s',
		async (fileType) => {
			mountWithSplat(fileType);
			await settle();

			expect(target.querySelector(`img[src="${SPLAT_URL}"]`)).toBeNull();
			expect(target.textContent).toContain('Loading scene');
			expect(viewerLoads).toEqual([SPLAT_URL]);
		}
	);

	it('shows the splat of a live generation from current_splat', async () => {
		mount({
			isGenerating: true,
			currentGeneration: {
				status: 'running',
				file_type: 'splat',
				current_splat: SPLAT_URL,
				splat_metadata: { format: 'ply', filename: '0.ply', splat_count: 1000 }
			}
		});
		await settle();

		expect(target.querySelector(`img[src="${SPLAT_URL}"]`)).toBeNull();
		expect(viewerLoads).toEqual([SPLAT_URL]);
	});

	it('reaches a splat that follows other outputs through the same absolute index', async () => {
		const MESH_URL = '/api/media/generations/gen-1/1.glb';
		mount({
			currentGeneration: { status: 'completed' },
			workbenchIndex: 1,
			workbenchTotal: 2,
			batchMeshes: [{ url: MESH_URL, originalUrl: MESH_URL, file_type: 'mesh', mesh_format: 'glb' }],
			batchSplats: [{ url: SPLAT_URL, originalUrl: SPLAT_URL, file_type: 'splat' }]
		});
		await settle();

		expect(viewerLoads).toEqual([SPLAT_URL]);
		expect(target.querySelector('model-viewer')).toBeNull();
	});

	it('lists the splat in the gallery strip when there is more than one output', async () => {
		mount({
			currentGeneration: { status: 'completed' },
			workbenchTotal: 2,
			batchSplats: [
				{ url: SPLAT_URL, originalUrl: SPLAT_URL, file_type: 'splat', splat_format: 'ply' },
				{
					url: '/api/media/generations/gen-1/1.spz',
					originalUrl: '/api/media/generations/gen-1/1.spz',
					file_type: 'splat',
					splat_format: 'spz'
				}
			]
		});
		await settle();

		const tiles = Array.from(target.querySelectorAll('[role="button"]')).filter((el) =>
			/^#[12]/.test((el.textContent || '').trim())
		);
		expect(tiles.map((el) => (el.textContent || '').replace(/\s+/g, ' ').trim())).toEqual([
			'#1 ply',
			'#2 spz'
		]);
	});
});
