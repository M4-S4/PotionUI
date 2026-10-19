import { describe, it, expect, beforeEach, vi } from 'vitest';
import { get } from 'svelte/store';
import { tabsStore } from '$lib/stores/tabs';
import { dispatchGenerationMessage } from '$lib/stores/generation';

// Importing '$lib/stores/generation' pulls in '$lib/generation/messages' as a
// side effect, which registers the workbench_update handler under test.

function defaultTabId(): string {
	return get(tabsStore).tabs[0].id;
}

function seedCurrentGeneration(tabId: string, generationId: string) {
	const tab = get(tabsStore).tabs.find((t) => t.id === tabId)!;
	tabsStore.updateTab(tabId, {
		generation: {
			...tab.generation,
			currentGeneration: { generation_id: generationId, status: 'running' }
		}
	});
}

function currentGeneration(tabId: string): any {
	return get(tabsStore).tabs.find((t) => t.id === tabId)!.generation.currentGeneration;
}

describe('workbench_update message handler - mesh', () => {
	beforeEach(() => tabsStore.reset());

	it('sets current_mesh and mesh_metadata for a final (non-temporary) mesh', () => {
		const tabId = defaultTabId();
		seedCurrentGeneration(tabId, 'gen-mesh-1');

		dispatchGenerationMessage(
			{
				type: 'workbench_update',
				generation_id: 'gen-mesh-1',
				pipe_id: 3,
				pipe_name: 'generator_mesh',
				output_type: 'mesh',
				file_type: 'mesh',
				mesh_format: 'glb',
				temporary: false,
				derived: false,
				seed: 4242,
				vertex_count: 3,
				face_count: 1,
				path: '/api/media/generations/gen-mesh-1/0.glb',
				mesh_name: '0.glb'
			} as any,
			{ unsubscribe: vi.fn() }
		);

		const gen = currentGeneration(tabId);
		expect(gen.current_mesh).toBe('/api/media/generations/gen-mesh-1/0.glb');
		expect(gen.file_type).toBe('mesh');
		expect(gen.current_image).toBeNull();
		expect(gen.current_video).toBeNull();
		expect(gen.current_audio).toBeNull();
		expect(gen.mesh_metadata).toEqual({
			format: 'glb',
			filename: '0.glb',
			vertex_count: 3,
			face_count: 1,
			seed: 4242,
			temporary: false,
			derived: false
		});
	});

	it('sets current_mesh from the tmp route for a temporary mesh', () => {
		const tabId = defaultTabId();
		seedCurrentGeneration(tabId, 'gen-mesh-2');

		dispatchGenerationMessage(
			{
				type: 'workbench_update',
				generation_id: 'gen-mesh-2',
				file_type: 'mesh',
				mesh_format: 'glb',
				temporary: true,
				path: '/api/media/tmp/gen-mesh-2/0.glb',
				mesh_name: '0.glb'
			} as any,
			{ unsubscribe: vi.fn() }
		);

		const gen = currentGeneration(tabId);
		expect(gen.current_mesh).toBe('/api/media/tmp/gen-mesh-2/0.glb');
		expect(gen.mesh_metadata.temporary).toBe(true);
	});

	it('ignores an unsaved mesh update that carries file_type but no path', () => {
		const tabId = defaultTabId();
		seedCurrentGeneration(tabId, 'gen-mesh-3');

		dispatchGenerationMessage(
			{
				type: 'workbench_update',
				generation_id: 'gen-mesh-3',
				file_type: 'mesh'
			} as any,
			{ unsubscribe: vi.fn() }
		);

		const gen = currentGeneration(tabId);
		expect(gen.current_mesh).toBeUndefined();
		expect(gen.file_type).toBeUndefined();
	});

	it('does not mistake a mesh update for a video update', () => {
		const tabId = defaultTabId();
		seedCurrentGeneration(tabId, 'gen-mesh-4');

		dispatchGenerationMessage(
			{
				type: 'workbench_update',
				generation_id: 'gen-mesh-4',
				file_type: 'mesh',
				path: '/api/media/generations/gen-mesh-4/0.glb'
			} as any,
			{ unsubscribe: vi.fn() }
		);

		const gen = currentGeneration(tabId);
		expect(gen.current_video).toBeNull();
		expect(gen.file_type).toBe('mesh');
	});
});

describe('workbench_update message handler - splat', () => {
	beforeEach(() => tabsStore.reset());

	it('sets current_splat and splat_metadata for a final (non-temporary) splat', () => {
		const tabId = defaultTabId();
		seedCurrentGeneration(tabId, 'gen-splat-1');

		dispatchGenerationMessage(
			{
				type: 'workbench_update',
				generation_id: 'gen-splat-1',
				pipe_id: 3,
				pipe_name: 'generator_splat',
				output_type: 'splat',
				file_type: 'splat',
				splat_format: 'ply',
				temporary: false,
				derived: false,
				seed: 4242,
				splat_count: 250000,
				path: '/api/media/generations/gen-splat-1/0.ply',
				splat_name: '0.ply'
			} as any,
			{ unsubscribe: vi.fn() }
		);

		const gen = currentGeneration(tabId);
		expect(gen.current_splat).toBe('/api/media/generations/gen-splat-1/0.ply');
		expect(gen.file_type).toBe('splat');
		expect(gen.current_image).toBeNull();
		expect(gen.current_video).toBeNull();
		expect(gen.current_audio).toBeNull();
		expect(gen.current_mesh).toBeUndefined();
		expect(gen.splat_metadata).toEqual({
			format: 'ply',
			filename: '0.ply',
			splat_count: 250000,
			seed: 4242,
			temporary: false,
			derived: false
		});
	});

	it('sets current_splat from the tmp route for a temporary splat', () => {
		const tabId = defaultTabId();
		seedCurrentGeneration(tabId, 'gen-splat-2');

		dispatchGenerationMessage(
			{
				type: 'workbench_update',
				generation_id: 'gen-splat-2',
				file_type: 'splat',
				splat_format: 'spz',
				temporary: true,
				path: '/api/media/tmp/gen-splat-2/0.spz'
			} as any,
			{ unsubscribe: vi.fn() }
		);

		const gen = currentGeneration(tabId);
		expect(gen.current_splat).toBe('/api/media/tmp/gen-splat-2/0.spz');
		expect(gen.splat_metadata.temporary).toBe(true);
		expect(gen.splat_metadata.format).toBe('spz');
	});

	it('ignores an unsaved splat update that carries file_type but no path', () => {
		const tabId = defaultTabId();
		seedCurrentGeneration(tabId, 'gen-splat-3');

		dispatchGenerationMessage(
			{
				type: 'workbench_update',
				generation_id: 'gen-splat-3',
				file_type: 'splat'
			} as any,
			{ unsubscribe: vi.fn() }
		);

		const gen = currentGeneration(tabId);
		expect(gen.current_splat).toBeUndefined();
		expect(gen.file_type).toBeUndefined();
	});

	it('does not mistake a splat update for a video update, nor for a mesh', () => {
		const tabId = defaultTabId();
		seedCurrentGeneration(tabId, 'gen-splat-4');

		dispatchGenerationMessage(
			{
				type: 'workbench_update',
				generation_id: 'gen-splat-4',
				file_type: 'splat',
				path: '/api/media/generations/gen-splat-4/0.ply'
			} as any,
			{ unsubscribe: vi.fn() }
		);

		const gen = currentGeneration(tabId);
		expect(gen.current_video).toBeNull();
		expect(gen.current_mesh).toBeUndefined();
		expect(gen.file_type).toBe('splat');
	});
});
