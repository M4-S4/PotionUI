import { beforeEach, describe, expect, it, vi } from 'vitest';
import { writable } from 'svelte/store';

const catalog = writable({ status: 'ready', items: [] as unknown[], ops: [] as unknown[], groups: [], error: null });
const loadFilterCatalog = vi.fn(() => Promise.resolve());
const loadCube = vi.fn();

vi.mock('./catalog', () => ({
	filterCatalog: catalog,
	loadFilterCatalog: () => loadFilterCatalog(),
	loadCube: (...args: unknown[]) => loadCube(...args)
}));

const { preselectLook } = await import('./preselect');

function item(id: string, extra: Record<string, unknown> = {}) {
	return {
		id,
		name: id.toUpperCase(),
		group: 'Colour',
		order: 0,
		intensity: 80,
		source: 'builtin',
		owned: false,
		has_lut: false,
		steps: [],
		unavailable_ops: [],
		needs_plugin: null,
		revision: '1',
		...extra
	};
}

function fakeSession() {
	return { closeFineTune: vi.fn(), clearFilter: vi.fn(), setFilter: vi.fn() };
}

beforeEach(() => {
	loadFilterCatalog.mockClear();
	loadCube.mockReset();
	catalog.set({ status: 'ready', items: [item('ember'), item('noir')], ops: [], groups: [], error: null });
});

describe('preselectLook', () => {
	it('selects the look in the editor session', async () => {
		const session = fakeSession();

		const done = await preselectLook(session as never, 'noir');

		expect(done).toBe(true);
		expect(session.setFilter).toHaveBeenCalledTimes(1);
		expect(session.setFilter.mock.calls[0][0]).toMatchObject({ id: 'noir', name: 'NOIR', defaultIntensity: 80 });
	});

	it('finds a plugin look by its whole id, colon and all, not by the part before the colon', async () => {
		const session = fakeSession();
		catalog.set({
			status: 'ready',
			items: [item('retro-tape'), item('retro-tape:vhs', { source: 'plugin', plugin_id: 'retro-tape' })],
			ops: [],
			groups: [],
			error: null
		});

		expect(await preselectLook(session as never, 'retro-tape:vhs')).toBe(true);

		expect(session.setFilter.mock.calls[0][0]).toMatchObject({ id: 'retro-tape:vhs', name: 'RETRO-TAPE:VHS' });
	});

	it('waits for the filter catalog first', async () => {
		const session = fakeSession();
		catalog.set({ status: 'idle', items: [], ops: [], groups: [], error: null });
		loadFilterCatalog.mockImplementationOnce(async () => {
			catalog.set({ status: 'ready', items: [item('ember')], ops: [], groups: [], error: null });
		});

		expect(await preselectLook(session as never, 'ember')).toBe(true);
		expect(loadFilterCatalog).toHaveBeenCalledTimes(1);
	});

	it('does nothing for a look that no longer exists', async () => {
		const session = fakeSession();

		expect(await preselectLook(session as never, 'gone')).toBe(false);
		expect(session.setFilter).not.toHaveBeenCalled();
	});

	it('does not select a look that is locked behind a missing plugin', async () => {
		const session = fakeSession();
		catalog.set({
			status: 'ready',
			items: [item('vhs', { needs_plugin: 'retro-tape' })],
			ops: [],
			groups: [],
			error: null
		});

		expect(await preselectLook(session as never, 'vhs')).toBe(false);
		expect(session.setFilter).not.toHaveBeenCalled();
	});

	it('reports failure rather than throwing when the look cannot be loaded', async () => {
		const session = fakeSession();
		catalog.set({ status: 'ready', items: [item('lut', { has_lut: true })], ops: [], groups: [], error: null });
		loadCube.mockRejectedValue(new Error('network'));

		expect(await preselectLook(session as never, 'lut')).toBe(false);
	});
});
