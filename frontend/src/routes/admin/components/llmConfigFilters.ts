import { parseServerDate } from '$lib/utils/relativeTime';
import { createFilterCodec, type FilterFieldDescriptor } from '$lib/components/library/filterCodec';
import type { SortOption } from '$lib/components/library/librarySection';

export type LLMConfigSortBy = 'name' | 'created';

export interface LLMConfigFilters {
	q: string;
	sortBy: LLMConfigSortBy;
}

export const DEFAULT_LLM_CONFIG_FILTERS: LLMConfigFilters = {
	q: '',
	sortBy: 'name'
};

export const LLM_CONFIG_SORT_OPTIONS: readonly SortOption<LLMConfigSortBy>[] = [
	{ value: 'name', label: 'Name' },
	{ value: 'created', label: 'Recently added' }
];

const FIELDS: readonly FilterFieldDescriptor<LLMConfigFilters>[] = [];

export const llmConfigFilters = createFilterCodec<LLMConfigFilters>({
	defaults: DEFAULT_LLM_CONFIG_FILTERS,
	fields: FIELDS,
	sortValues: ['name', 'created']
});

function configTimestamp(config: any): number {
	return parseServerDate(config?.created_at)?.getTime() ?? 0;
}

export function applyLLMConfigFilters(configurations: readonly any[], filters: LLMConfigFilters): any[] {
	const query = filters.q.trim().toLowerCase();
	const rows = configurations.filter((config) => {
		if (!query) return true;
		return (config.name ?? '').toLowerCase().includes(query) || (config.model ?? '').toLowerCase().includes(query);
	});
	if (filters.sortBy === 'created') return rows.sort((a, b) => configTimestamp(b) - configTimestamp(a));
	return rows.sort((a, b) => (a.name ?? '').localeCompare(b.name ?? ''));
}
