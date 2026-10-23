import type { Prompt } from '$lib/types/segments';
import type { APIResponse } from '$lib/types/api';
import type { PromptListParams } from '$lib/services/api/prompts';
import { promptFiltersToListParams, type PromptFilters } from './promptFilters';

export interface PromptLibraryApi {
	listPrompts(
		params: PromptListParams
	): Promise<APIResponse<{ items: Prompt[]; total: number; semantic_hits?: number }>>;
}

export interface PromptLibraryQueryOptions {
	collectionId?: string;
	unsorted?: boolean;
	directOnly?: boolean;
	favoritesOnly?: boolean;
	limit: number;
	offset: number;
}

export interface PromptLibraryQueryResult {
	rows: Prompt[];
	total: number;
	semanticHits: number;
}

export async function queryPromptLibrary(
	apiLike: PromptLibraryApi,
	filters: PromptFilters,
	options: PromptLibraryQueryOptions
): Promise<PromptLibraryQueryResult> {
	const query = filters.q.trim();
	const response = await apiLike.listPrompts({
		...promptFiltersToListParams(filters, options),
		q: query || undefined,
		semantic: query ? true : undefined
	});
	if (!response.success) return { rows: [], total: 0, semanticHits: 0 };
	return {
		rows: response.data?.items || [],
		total: response.data?.total ?? 0,
		semanticHits: response.data?.semantic_hits ?? 0
	};
}
