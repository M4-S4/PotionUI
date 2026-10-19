import { beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';
import { dispatchGenerationMessage } from '$lib/stores/generation';
import { tabsStore } from '$lib/stores/tabs';
import {
	leadOutputPatch,
	peekGenerationOutputs,
	resetGenerationOutputsRetirementForTests
} from './generationOutputs';

let ID = '';
let counter = 0;
let SPLAT_PATH = '';

function start() {
	const tab = get(tabsStore).tabs[0];
	tabsStore.updateTab(tab.id, {
		generation: {
			...tab.generation,
			isGenerating: true,
			currentGeneration: { id: ID, generation_id: ID, status: 'running' }
		}
	});
	return tab.id;
}

function send(payload: Record<string, unknown>) {
	dispatchGenerationMessage(
		{ type: 'gallery_update', generation_id: ID, images: [], videos: [], audios: [], ...payload } as any,
		{ unsubscribe: vi.fn() }
	);
}

function complete() {
	dispatchGenerationMessage(
		{ type: 'generation_complete', generation_id: ID, data: { status: 'completed' } } as any,
		{ unsubscribe: vi.fn() }
	);
}

function generation(tabId: string) {
	return get(tabsStore).tabs.find((t) => t.id === tabId)!.generation;
}

describe('gallery_update splats', () => {
	beforeEach(() => {
		tabsStore.reset();
		resetGenerationOutputsRetirementForTests();
		ID = `gen-splat-${++counter}`;
		SPLAT_PATH = `/api/media/generations/${ID}/0.ply`;
	});

	it('builds batchSplats from splats and splat_urls_list, and counts them in workbenchTotal', () => {
		const tabId = start();
		send({
			splats: [{ file_type: 'splat', splat_format: 'ply', splat_count: 250000, seed: 3, temporary: false }],
			splat_urls_list: [{ file_type: 'splat', splat_format: 'ply', splat_count: 250000, seed: 3, path: SPLAT_PATH }]
		});

		const gen = generation(tabId);
		expect(gen.batchSplats).toEqual([
			{
				url: SPLAT_PATH,
				originalUrl: SPLAT_PATH,
				file_type: 'splat',
				splat_format: 'ply',
				derived: false,
				splat_count: 250000,
				seed: 3
			}
		]);
		expect(gen.batchMeshes).toEqual([]);
		expect(gen.workbenchTotal).toBe(1);
	});

	it('skips a temporary splat that has no path instead of mounting an empty viewer', () => {
		const tabId = start();
		send({ splats: [{ file_type: 'splat', splat_format: 'ply', temporary: true }], splat_urls_list: [] });

		expect(generation(tabId).batchSplats).toEqual([]);
		expect(generation(tabId).workbenchTotal).toBe(0);
	});

	it('keeps the meshes of the same generation when a later update carries only splats', () => {
		const tabId = start();
		send({
			meshes: [{ file_type: 'mesh', mesh_format: 'glb' }],
			mesh_urls_list: [{ path: `/api/media/generations/${ID}/1.glb` }]
		});
		send({
			splats: [{ file_type: 'splat', splat_format: 'spz', derived: true }],
			splat_urls_list: [{ path: `/api/media/generations/${ID}/0.spz`, derived: true }]
		});

		const gen = generation(tabId);
		expect(gen.batchMeshes).toHaveLength(1);
		expect(gen.batchSplats).toHaveLength(1);
		expect(gen.batchSplats?.[0].derived).toBe(true);
		expect(gen.workbenchTotal).toBe(2);
		expect(peekGenerationOutputs(ID).splats).toHaveLength(1);
	});

	it('shows the splat when the generation completes, and not as a mesh', () => {
		const tabId = start();
		send({
			splats: [{ file_type: 'splat', splat_format: 'ply' }],
			splat_urls_list: [{ path: SPLAT_PATH }]
		});
		complete();

		const gen = generation(tabId);
		expect(gen.currentGeneration?.status).toBe('completed');
		expect(gen.currentGeneration?.file_type).toBe('splat');
		expect(gen.currentGeneration?.current_splat).toBe(SPLAT_PATH);
		expect(gen.currentGeneration?.current_mesh).toBeNull();
		expect(gen.batchSplats).toHaveLength(1);
		expect(gen.workbenchIndex).toBe(0);
		expect(gen.workbenchTotal).toBe(1);
	});
});

describe('leadOutputPatch with splats', () => {
	const image = { url: '/i.png', originalUrl: '/i.png' };
	const mesh = { url: '/m.glb', originalUrl: '/m.glb' };
	const splat = { url: '/s.ply', originalUrl: '/s.ply' };
	const empty = { images: [], videos: [], audios: [], meshes: [], splats: [] };

	it('leads with the splat when it is the only output', () => {
		const patch = leadOutputPatch({ ...empty, splats: [splat] });
		expect(patch.file_type).toBe('splat');
		expect(patch.current_splat).toBe('/s.ply');
		expect(patch.current_mesh).toBeNull();
		expect(patch.batchSplats).toEqual([splat]);
		expect(patch.workbenchIndex).toBe(0);
		expect(patch.workbenchTotal).toBe(1);
	});

	it('leads with the first item of the chain and keeps splats last', () => {
		const patch = leadOutputPatch({ ...empty, images: [image], meshes: [mesh], splats: [splat] });
		expect(patch.file_type).toBe('image');
		expect(patch.workbenchTotal).toBe(3);
		expect(patch.current_splat).toBeNull();
	});

	it('leads with a derived splat, reading the file type from the splat bucket', () => {
		const derived = { ...splat, derived: true };
		const patch = leadOutputPatch({ ...empty, meshes: [mesh], splats: [splat, derived] });
		expect(patch.workbenchIndex).toBe(2);
		expect(patch.file_type).toBe('splat');
		expect(patch.current_splat).toBe('/s.ply');
	});

	it('reports no file type for an empty generation', () => {
		const patch = leadOutputPatch(empty);
		expect(patch.file_type).toBeNull();
		expect(patch.current_splat).toBeNull();
	});
});
