import { describe, it, expect, vi, beforeEach } from 'vitest';

const getGenerationById = vi.fn();
const updateGenerationTags = vi.fn();

vi.mock('$lib/services/api/index', () => ({
	api: {
		getGenerationById: (...args: unknown[]) => getGenerationById(...args),
		updateGenerationTags: (...args: unknown[]) => updateGenerationTags(...args)
	}
}));

const { fetchGenerationTags, saveGenerationTags, tagChips } = await import('./generationTags');

beforeEach(() => {
	getGenerationById.mockReset();
	updateGenerationTags.mockReset();
});

describe('tagChips', () => {
	it('keeps only id and name and tolerates a missing list', () => {
		expect(tagChips([{ id: 't1', name: 'one', color: 'red' } as { id: string; name: string }])).toEqual([{ id: 't1', name: 'one' }]);
		expect(tagChips(undefined)).toEqual([]);
	});
});

describe('fetchGenerationTags', () => {
	it('asks for tags without files and returns chips', async () => {
		getGenerationById.mockResolvedValue({ success: true, data: { tags: [{ id: 't1', name: 'one' }] } });
		expect(await fetchGenerationTags('g1')).toEqual([{ id: 't1', name: 'one' }]);
		expect(getGenerationById).toHaveBeenCalledWith('g1', true, false);
	});

	it('returns null when the response carries no tags', async () => {
		getGenerationById.mockResolvedValue({ success: true, data: {} });
		expect(await fetchGenerationTags('g1')).toBeNull();
	});
});

describe('saveGenerationTags', () => {
	it('returns the saved tags from the response', async () => {
		const tags = [{ id: 't2', name: 'two', color: 'blue' }];
		updateGenerationTags.mockResolvedValue({ success: true, data: { tags } });
		expect(await saveGenerationTags('g1', ['t2'])).toBe(tags);
		expect(updateGenerationTags).toHaveBeenCalledWith('g1', ['t2']);
	});

	it('returns null on failure', async () => {
		updateGenerationTags.mockResolvedValue({ success: false, message: 'no' });
		expect(await saveGenerationTags('g1', [])).toBeNull();
	});
});
