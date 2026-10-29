import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services/api/mediaTools', () => ({ listMediaTools: vi.fn() }));

import { buildLibraryToolContext, unregisterMediaTools, type MediaToolItem } from './tools';
import { registerEntryTools } from './entryTools';
import { syncCatalogTools } from './catalogTools';
import { buildRowContext, buildRowModel, type RowTool } from './rowModel';
import type { LibraryItem } from '$lib/services/api/library';
import { A1111_TOOL, BOOMERANG_TOOL, CORE_TOOLS } from '../../../tests/component/fixtures/mediaToolCatalog';

registerEntryTools();

function item(kind: MediaToolItem['kind'], extra: Partial<MediaToolItem> = {}): MediaToolItem {
	return { id: 'i1', kind, url: `/u/i1.${kind}`, filename: `clip.${kind === 'image' ? 'png' : 'bin'}`, ...extra };
}

function ids(entries: RowTool[]): string[] {
	return entries.map((entry) => entry.id);
}

beforeEach(() => {
	unregisterMediaTools((tool) => !!tool.catalogToolId);
	syncCatalogTools([...CORE_TOOLS, A1111_TOOL, BOOMERANG_TOOL]);
});

describe('buildRowModel', () => {
	it('puts tools with a built-in editor on buttons and the rest under More tools, for a video', () => {
		const model = buildRowModel(buildRowContext('library', [item('video')]));
		expect(ids(model.editButtons)).toEqual(['trim', 'crop', 'resize', 'frame']);
		expect(ids(model.moreTools)).toEqual(['rotate', 'flip', 'speed', 'boomerang.loop']);
	});

	it('adds Edit image after the editor tools for an image', () => {
		const model = buildRowModel(buildRowContext('library', [item('image')]));
		expect(ids(model.editButtons)).toEqual(['crop', 'resize', 'edit-image']);
		expect(ids(model.moreTools)).toEqual(['rotate', 'flip', 'fit_canvas', 'color_key', 'crop_subject']);
	});

	it('offers audio trim and split as buttons and speed under More tools', () => {
		const model = buildRowModel(buildRowContext('library', [item('audio')]));
		expect(ids(model.editButtons)).toEqual(['trim', 'split']);
		expect(ids(model.moreTools)).toEqual(['speed']);
	});

	it('shows Effects for an image only', () => {
		expect(ids(buildRowModel(buildRowContext('library', [item('image')])).effects)).toEqual(['apply-look']);
		expect(buildRowModel(buildRowContext('library', [item('video')])).effects).toEqual([]);
		expect(buildRowModel(buildRowContext('library', [item('audio')])).effects).toEqual([]);
	});

	it('hides a tool that needs more than one item instead of showing it disabled', () => {
		const model = buildRowModel(buildRowContext('library', [item('video')]));
		expect(ids([...model.editButtons, ...model.moreTools])).not.toContain('concat');
	});

	it('keeps a tool the item cannot take visible, disabled, with its reason', () => {
		const model = buildRowModel(buildRowContext('library', [item('image', { filename: 'photo.jpg' })]));
		const subject = model.moreTools.find((entry) => entry.id === 'crop_subject');
		expect(subject).toMatchObject({ enabled: false, reason: 'Needs a transparent image' });
	});

	it('leaves out the A1111 export, which stays in the media corner', () => {
		const model = buildRowModel(buildRowContext('history', [item('image', { generationId: 'g' })]));
		expect(ids([...model.editButtons, ...model.moreTools, ...model.effects])).not.toContain(A1111_TOOL.id);
	});

	it('gives both pages the same Edit and Effects for the same kind of media', () => {
		for (const kind of ['image', 'video', 'audio'] as const) {
			const library = buildRowModel(buildRowContext('library', [item(kind)]));
			const history = buildRowModel(buildRowContext('history', [item(kind, { generationId: 'g' })]));
			expect(ids(history.editButtons)).toEqual(ids(library.editButtons));
			expect(ids(history.moreTools)).toEqual(ids(library.moreTools));
			expect(ids(history.effects)).toEqual(ids(library.effects));
		}
	});

	it('is empty for media nothing edits', () => {
		const model = buildRowModel(buildRowContext('history', [item('mesh')]));
		expect(model).toEqual({ editButtons: [], moreTools: [], effects: [] });
	});

	it('carries the note a tool shows under its label', () => {
		const model = buildRowModel(buildRowContext('library', [item('video')]));
		expect(model.editButtons.find((entry) => entry.id === 'frame')?.note).toBe('Saves one image');
	});
});

describe('recent looks in the Effects row', () => {
	const looks = [
		{ id: 'noir', name: 'Noir', group: 'Black and white' },
		{ id: 'retro-tape:vhs', name: 'VHS', group: 'Retro Tape' }
	];

	it('follow Apply a look, one button each, in the order given', () => {
		const model = buildRowModel(buildRowContext('library', [item('image')]), looks);

		expect(ids(model.effects)).toEqual(['apply-look', 'look:noir', 'look:retro-tape:vhs']);
		expect(model.effects.map((entry) => entry.label)).toEqual(['Apply a look…', 'Noir', 'VHS']);
	});

	it('carry the look id and group, which Apply a look does not', () => {
		const [open, noir] = buildRowModel(buildRowContext('library', [item('image')]), looks).effects;

		expect(open.lookId).toBeUndefined();
		expect(noir).toMatchObject({ lookId: 'noir', note: 'Black and white', action: 'apply-look' });
	});

	it('are left out for video and audio along with the rest of Effects', () => {
		expect(buildRowModel(buildRowContext('library', [item('video')]), looks).effects).toEqual([]);
		expect(buildRowModel(buildRowContext('library', [item('audio')]), looks).effects).toEqual([]);
	});

	it('show on both pages alike', () => {
		const library = buildRowModel(buildRowContext('library', [item('image')]), looks);
		const history = buildRowModel(buildRowContext('history', [item('image', { generationId: 'g' })]), looks);

		expect(ids(history.effects)).toEqual(ids(library.effects));
	});

	it('are absent when no look was applied yet', () => {
		expect(ids(buildRowModel(buildRowContext('library', [item('image')])).effects)).toEqual(['apply-look']);
	});
});

describe('buildRowContext', () => {
	it('builds the same context a card entry does, so hidden rules agree', () => {
		const ctx = buildRowContext('library', [item('video')]);
		expect(ctx.scope).toBe('library');
		expect([...ctx.kinds]).toEqual(['video']);
		expect(ctx.entry?.status).toBe('completed');
	});

	it('agrees with the library context on the items it is given', () => {
		const row = { id: 'u1', filename: 'u1.png', media_type: 'image', url: '/u/u1.png', tags: [] } as LibraryItem;
		const ctx = buildLibraryToolContext([row], null);
		expect(buildRowContext('library', ctx.items).items).toEqual(ctx.items);
	});
});
