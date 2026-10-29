import { describe, it, expect, vi, beforeEach } from 'vitest';

const listGenerationMedia = vi.fn();
const listUploads = vi.fn();
vi.mock('$lib/services/api/index', () => ({
	api: {
		listGenerationMedia: (...args: unknown[]) => listGenerationMedia(...args),
		listUploads: (...args: unknown[]) => listUploads(...args)
	}
}));

import { canReplace, NOT_IN_LIBRARY, normalizeSources, resolveToolInput, resolveToolInputs } from './inputs';
import type { MediaToolSource } from './types';

function source(overrides: Partial<MediaToolSource> = {}): MediaToolSource {
	return { id: 'x', kind: 'image', url: '/a.png', filename: 'a.png', ...overrides };
}

beforeEach(() => {
	listGenerationMedia.mockReset();
	listUploads.mockReset();
	listGenerationMedia.mockResolvedValue({ success: false });
	listUploads.mockResolvedValue({ success: true, data: { uploads: [], total: 0, limit: 100, offset: 0 } });
});

describe('normalizeSources', () => {
	it('takes a library item for the row itself', () => {
		expect(normalizeSources([source({ id: 'row-1' })], 'library')[0].itemId).toBe('row-1');
	});

	it('leaves a history item alone, since its id names a generation', () => {
		const [out] = normalizeSources([source({ id: 'g1:0', generationId: 'g1' })], 'history');
		expect(out.itemId).toBeUndefined();
	});

	it('does not invent a row for a field item', () => {
		expect(normalizeSources([source({ id: 'field' })], 'field')[0].itemId).toBeUndefined();
	});

	it('keeps a row id the caller already gave', () => {
		expect(normalizeSources([source({ id: 'x', itemId: 'row-7' })], 'library')[0].itemId).toBe('row-7');
	});
});

describe('resolveToolInput', () => {
	it('sends a known generation file as it is', async () => {
		expect(await resolveToolInput(source({ generationFileId: 'file-1', itemId: 'row-1' }))).toEqual({
			ref: { generation_file_id: 'file-1' },
			reason: null
		});
	});

	it('sends a library row as an item id', async () => {
		expect(await resolveToolInput(source({ itemId: 'row-1' }))).toEqual({ ref: { item_id: 'row-1' }, reason: null });
	});

	it('finds the generation file of a history item by the file name in its url', async () => {
		listGenerationMedia.mockResolvedValue({
			success: true,
			data: { media: [{ id: 'f0', filename: '0.png' }, { id: 'f1', filename: '1.png' }] }
		});
		const result = await resolveToolInput(
			source({ id: 'gen-9:1', generationId: 'gen-9', url: '/api/media/generations/gen-9/1.png', filename: 'shown name.png' })
		);
		expect(listGenerationMedia).toHaveBeenCalledWith('gen-9');
		expect(result).toEqual({ ref: { generation_file_id: 'f1' }, reason: null });
	});

	it('says so when a history file cannot be found', async () => {
		const result = await resolveToolInput(source({ generationId: 'gen-9', url: '/api/media/generations/gen-9/1.png' }));
		expect(result).toEqual({ ref: null, reason: NOT_IN_LIBRARY });
	});

	it('finds the library row of a field value from its stored path', async () => {
		listUploads.mockResolvedValue({
			success: true,
			data: { uploads: [{ id: 'row-77', filename: 'clip.mp4' }], total: 1, limit: 100, offset: 0 }
		});
		const result = await resolveToolInput(source({ kind: 'video', storedPath: 'uploads/clip.mp4' }));
		expect(result).toEqual({ ref: { item_id: 'row-77' }, reason: null });
	});

	it('treats a generated file in a field as a generation file, with no copy into the library', async () => {
		listGenerationMedia.mockResolvedValue({
			success: true,
			data: { media: [{ id: 'f9', filename: '0.mp4' }] }
		});
		const result = await resolveToolInput(source({ kind: 'video', storedPath: 'outputs/2026-08-13/01KABC/0.mp4' }));
		expect(result).toEqual({ ref: { generation_file_id: 'f9' }, reason: null });
	});

	it('gives a reason when nothing can be found', async () => {
		const result = await resolveToolInput(source({ storedPath: 'somewhere/else.png' }));
		expect(result.ref).toBeNull();
		expect(result.reason).toContain('not in your library');
	});

	it('does not try to edit a mesh', async () => {
		expect((await resolveToolInput(source({ kind: 'mesh' }))).ref).toBeNull();
	});
});

describe('resolveToolInputs', () => {
	it('resolves every source in order', async () => {
		const result = await resolveToolInputs([source({ itemId: 'a' }), source({ itemId: 'b' }), source({ generationFileId: 'c' })]);
		expect(result).toEqual({ refs: [{ item_id: 'a' }, { item_id: 'b' }, { generation_file_id: 'c' }], reason: null });
	});

	it('fails the lot when one cannot be resolved', async () => {
		const result = await resolveToolInputs([source({ itemId: 'a' }), source({ storedPath: 'nowhere/x.png' })]);
		expect(result.refs).toEqual([]);
		expect(result.reason).toContain('not in your library');
	});
});

describe('canReplace', () => {
	it('allows replacing one library row only', () => {
		expect(canReplace([{ item_id: 'a' }])).toBe(true);
		expect(canReplace([{ generation_file_id: 'a' }])).toBe(false);
		expect(canReplace([{ item_id: 'a' }, { item_id: 'b' }])).toBe(false);
		expect(canReplace([])).toBe(false);
	});
});
