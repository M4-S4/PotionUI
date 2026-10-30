import { beforeEach, describe, expect, it, vi } from 'vitest';

const openMediaTool = vi.fn();
vi.mock('$lib/media-tools/open', () => ({ openMediaTool: (...args: unknown[]) => openMediaTool(...args) }));

import { pickTool } from './toolPicker';
import { buildHistoryToolContext, buildLibraryToolContext, type MediaTool } from './tools';
import { mediaToolFromCatalog } from './catalogTools';
import { A1111_TOOL, CORE_TOOLS } from '../../../tests/component/fixtures/mediaToolCatalog';

const coreModal: MediaTool = {
	id: 'compare',
	label: 'Compare',
	icon: 'layers',
	category: 'analyze',
	source: 'core',
	scopes: ['history', 'library'],
	applies: () => ({ enabled: true })
};

const libraryCtx = buildLibraryToolContext(
	[{ id: 'u1', filename: 'u1.png', media_type: 'image', url: '/u/u1.png', tags: [] } as never],
	null
);

const handlers = () => ({ mount: vi.fn(), onResult: vi.fn() });

beforeEach(() => {
	openMediaTool.mockReset();
});

describe('pickTool', () => {
	it('mounts a core tool through the page host and does not open the media tool dialog', () => {
		const h = handlers();

		pickTool(coreModal, libraryCtx, h);

		expect(h.mount).toHaveBeenCalledWith(coreModal, libraryCtx);
		expect(openMediaTool).not.toHaveBeenCalled();
	});

	it('opens a catalog tool on the selection, in the scope and context of the page', () => {
		const h = handlers();

		pickTool(mediaToolFromCatalog(CORE_TOOLS[1]), libraryCtx, h);

		expect(h.mount).not.toHaveBeenCalled();
		expect(openMediaTool).toHaveBeenCalledTimes(1);
		const call = openMediaTool.mock.calls[0][0];
		expect(call.toolId).toBe('crop');
		expect(call.scope).toBe('library');
		expect(call.items).toBe(libraryCtx.items);
		expect(call.context).toBe(libraryCtx);
	});

	it('opens a component-only plugin tool the same way, with the History context it needs', () => {
		const h = handlers();
		const ctx = buildHistoryToolContext([], [], null);

		pickTool(mediaToolFromCatalog(A1111_TOOL), ctx, h);

		expect(openMediaTool.mock.calls[0][0]).toMatchObject({ toolId: A1111_TOOL.id, scope: 'history' });
		expect(openMediaTool.mock.calls[0][0].context).toBe(ctx);
	});

	it('hands what the tool produced to the page together with the selection it ran on', async () => {
		const h = handlers();
		pickTool(mediaToolFromCatalog(CORE_TOOLS[1]), libraryCtx, h);

		const produced = [{ id: 'new-1' }];
		await openMediaTool.mock.calls[0][0].onResult(produced, { toolId: 'crop', replaced: false });

		expect(h.onResult).toHaveBeenCalledWith(produced, libraryCtx, { toolId: 'crop', replaced: false });
	});

	it('tells the page whether the run replaced the original', async () => {
		const h = handlers();
		pickTool(mediaToolFromCatalog(CORE_TOOLS[1]), libraryCtx, h);

		await openMediaTool.mock.calls[0][0].onResult([{ id: 'u1' }], { toolId: 'crop', replaced: true });

		expect(h.onResult.mock.calls[0][2]).toEqual({ toolId: 'crop', replaced: true });
	});
});
