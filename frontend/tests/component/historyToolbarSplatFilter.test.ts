// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('$lib/services/api/index', () => ({
	api: {
		getModels: vi.fn(),
		getModelTypes: vi.fn(),
		getGenerationHistory: vi.fn(),
		getHistoryFacets: vi.fn(),
		searchPhrasebook: vi.fn(),
		setOnAuthExpired: vi.fn(),
		setOnTokenRefreshed: vi.fn(),
		listPresets: vi.fn().mockResolvedValue({ success: true, data: [] })
	}
}));

const { api } = await import('$lib/services/api/index');
const { historyStore } = await import('$lib/stores/history');
const { default: HistoryToolbar } = await import('../../src/routes/history/components/HistoryToolbar.svelte');
const { createClassComponent } = await import('svelte/legacy');
const { iconPaths } = await import('$lib/utils/IconLibrary');

async function settle() {
	for (let i = 0; i < 8; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

function filterButton(label: string) {
	return document.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);
}

describe('History media type filter', () => {
	let destroy: () => void;

	beforeEach(() => {
		vi.mocked(api.getGenerationHistory).mockResolvedValue({
			success: true,
			data: { generations: [], total: 0 }
		} as never);
		vi.mocked(api.getHistoryFacets).mockResolvedValue({
			success: true,
			data: { modes: [], presets: [], models: [] }
		} as never);
		historyStore.setFilter('mediaType', 'all');
		const target = document.createElement('div');
		document.body.appendChild(target);
		const component = createClassComponent({
			component: HistoryToolbar as never,
			target,
			props: { onOpenUpload: () => {}, onOpenAddTag: () => {}, onOpenDeleteByCriteria: () => {} }
		});
		destroy = () => {
			component.$destroy();
			target.remove();
		};
	});

	afterEach(() => {
		destroy();
		historyStore.setFilter('mediaType', 'all');
		document.body.innerHTML = '';
		vi.mocked(api.getGenerationHistory).mockClear();
	});

	it('offers a splat filter next to the mesh one, with the splat icon', () => {
		const splat = filterButton('3D scenes only');
		const mesh = filterButton('3D only');

		expect(splat).not.toBeNull();
		expect(mesh).not.toBeNull();
		expect(splat).not.toBe(mesh);
		expect(splat!.querySelector('svg path')?.getAttribute('d')).toBe((iconPaths.splat as string[])[0]);
	});

	it('asks the server for splat files only when the splat filter is picked', async () => {
		filterButton('3D scenes only')!.click();
		await settle();

		expect(vi.mocked(api.getGenerationHistory)).toHaveBeenLastCalledWith(
			expect.objectContaining({ mediaType: 'splat' })
		);
		expect(filterButton('3D scenes only')!.className).toContain('text-signal');
		expect(filterButton('3D only')!.className).not.toContain('text-signal');
	});

	it('still asks for meshes only when the mesh filter is picked', async () => {
		filterButton('3D only')!.click();
		await settle();

		expect(vi.mocked(api.getGenerationHistory)).toHaveBeenLastCalledWith(
			expect.objectContaining({ mediaType: 'mesh' })
		);
	});
});
