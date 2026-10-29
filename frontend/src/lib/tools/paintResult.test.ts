import { beforeEach, describe, expect, it, vi } from 'vitest';

const applyToolResult = vi.hoisted(() => vi.fn());
const success = vi.hoisted(() => vi.fn());

vi.mock('$lib/stores/library', () => ({ libraryStore: { applyToolResult: (...args: unknown[]) => applyToolResult(...args) } }));
vi.mock('$lib/stores/toast', () => ({ toasts: { success: (...args: unknown[]) => success(...args) } }));

import { applyLibraryPaintResult, PAINT_REPLACED, PAINT_SAVED_AS_NEW } from './paintResult';

const item = { id: 'row-1', filename: 'x.png', media_type: 'image', url: '/x.png' };

beforeEach(() => {
	applyToolResult.mockReset().mockResolvedValue({ added: 0, replaced: 0 });
	success.mockReset();
});

describe('applyLibraryPaintResult', () => {
	it('adds a new image next to its source and says it was saved as new', async () => {
		await applyLibraryPaintResult({ type: 'item', item, replaced: false } as never, ['row-1']);

		expect(applyToolResult).toHaveBeenCalledWith([item], ['row-1'], { replaced: false });
		expect(success).toHaveBeenCalledWith(PAINT_SAVED_AS_NEW);
	});

	it('patches a replaced image and says the tags and collections were kept', async () => {
		await applyLibraryPaintResult({ type: 'item', item, replaced: true } as never, ['row-1']);

		expect(applyToolResult).toHaveBeenCalledWith([item], ['row-1'], { replaced: true });
		expect(success).toHaveBeenCalledWith(PAINT_REPLACED);
	});

	it('adds every image of a result that holds several, never as a replace', async () => {
		await applyLibraryPaintResult({ type: 'items', items: [item, { ...item, id: 'row-2' }] } as never, ['row-1']);

		expect(applyToolResult.mock.calls[0][0]).toHaveLength(2);
		expect(applyToolResult.mock.calls[0][2]).toEqual({ replaced: false });
	});

	it('waits for the store before it says anything', async () => {
		const order: string[] = [];
		applyToolResult.mockImplementation(async () => void order.push('store'));
		success.mockImplementation(() => void order.push('toast'));

		await applyLibraryPaintResult({ type: 'item', item, replaced: false } as never, ['row-1']);

		expect(order).toEqual(['store', 'toast']);
	});

	it('ignores a result that is not an image, such as a mask path', async () => {
		await applyLibraryPaintResult({ type: 'mask', maskPath: 'x' } as never, ['row-1']);

		expect(applyToolResult).not.toHaveBeenCalled();
		expect(success).not.toHaveBeenCalled();
	});
});
