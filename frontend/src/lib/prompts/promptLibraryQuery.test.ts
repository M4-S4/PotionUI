import { describe, it, expect, vi } from 'vitest';
import type { Prompt } from '$lib/types/segments';
import { DEFAULT_PROMPT_FILTERS } from './promptFilters';
import { queryPromptLibrary } from './promptLibraryQuery';

function prompt(id: string): Prompt {
	return { id, display_name: id, segments: [], flattened_text: id };
}

function fakeApi(items: Prompt[], total: number, semanticHits?: number) {
	return {
		listPrompts: vi.fn(async () => ({ success: true, data: { items, total, semantic_hits: semanticHits } }))
	};
}

describe('queryPromptLibrary', () => {
	it('asks the list endpoint for semantic matches alongside the text query and the active filters', async () => {
		const api = fakeApi([prompt('a'), prompt('b')], 2, 1);
		const result = await queryPromptLibrary(
			api,
			{ ...DEFAULT_PROMPT_FILTERS, q: ' women ', hasVariables: true },
			{ limit: 48, offset: 0 }
		);
		expect(api.listPrompts).toHaveBeenCalledWith(
			expect.objectContaining({ q: 'women', semantic: true, has_variables: true, limit: 48, offset: 0 })
		);
		expect(result).toEqual({ rows: [prompt('a'), prompt('b')], total: 2, semanticHits: 1 });
	});

	it('pages with the same query and offset', async () => {
		const api = fakeApi([prompt('d')], 49);
		const result = await queryPromptLibrary(api, { ...DEFAULT_PROMPT_FILTERS, q: 'dancer' }, { limit: 48, offset: 48 });
		expect(api.listPrompts).toHaveBeenCalledWith(expect.objectContaining({ q: 'dancer', semantic: true, offset: 48 }));
		expect(result).toEqual({ rows: [prompt('d')], total: 49, semanticHits: 0 });
	});

	it('skips semantic matching without a query and passes the filters through as list params', async () => {
		const api = fakeApi([prompt('a')], 1);
		const result = await queryPromptLibrary(
			api,
			{ ...DEFAULT_PROMPT_FILTERS, source: 'civitai', showNsfw: true },
			{ collectionId: 'col-1', limit: 48, offset: 0 }
		);
		expect(api.listPrompts).toHaveBeenCalledWith(
			expect.objectContaining({ q: undefined, semantic: undefined, collection_id: 'col-1', source_provider: 'civitai', nsfw: 'include' })
		);
		expect(result).toEqual({ rows: [prompt('a')], total: 1, semanticHits: 0 });
	});

	it('returns an empty page with a zero total when the list call fails', async () => {
		const api = { listPrompts: vi.fn(async () => ({ success: false as const, error: 'boom' })) };
		const result = await queryPromptLibrary(api, DEFAULT_PROMPT_FILTERS, { limit: 48, offset: 0 });
		expect(result).toEqual({ rows: [], total: 0, semanticHits: 0 });
	});
});
