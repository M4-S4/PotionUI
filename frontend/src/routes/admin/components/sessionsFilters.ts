import { createFilterCodec } from '$lib/components/library/filterCodec';
import type { SortOption } from '$lib/components/library/librarySection';

export type SessionsSortBy = 'recent';

export interface SessionsFilters {
	q: string;
	sortBy: SessionsSortBy;
}

export const DEFAULT_SESSIONS_FILTERS: SessionsFilters = {
	q: '',
	sortBy: 'recent'
};

export const SESSIONS_SORT_OPTIONS: readonly SortOption<SessionsSortBy>[] = [
	{ value: 'recent', label: 'Most recent' }
];

export const sessionsFilters = createFilterCodec<SessionsFilters>({
	defaults: DEFAULT_SESSIONS_FILTERS,
	fields: [],
	sortValues: ['recent']
});
