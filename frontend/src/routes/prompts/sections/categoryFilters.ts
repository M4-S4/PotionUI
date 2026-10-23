import type { SegmentCategory } from '$lib/types/segments';
import { createFilterCodec, type FilterFieldDescriptor } from '$lib/components/library/filterCodec';

export type CategoryHasFilter = '' | 'segments' | 'empty';
export type CategorySortBy = 'name' | 'count';

export interface CategoryFilters {
	q: string;
	has: CategoryHasFilter;
	sortBy: CategorySortBy;
}

export const DEFAULT_CATEGORY_FILTERS: CategoryFilters = {
	q: '',
	has: '',
	sortBy: 'name'
};

export const CATEGORY_SORT_OPTIONS: ReadonlyArray<{ value: CategorySortBy; label: string }> = [
	{ value: 'name', label: 'Name' },
	{ value: 'count', label: 'Segment count' }
];

const FIELDS: readonly FilterFieldDescriptor<CategoryFilters>[] = [
	{
		kind: 'enum',
		key: 'has',
		param: 'has',
		label: 'Segments',
		values: ['segments', 'empty'],
		default: '',
		chipLabel: (value) => (value === 'segments' ? 'with segments' : 'empty')
	}
];

export const categoryFilters = createFilterCodec<CategoryFilters>({
	defaults: DEFAULT_CATEGORY_FILTERS,
	fields: FIELDS,
	sortValues: ['name', 'count']
});

function byName(a: SegmentCategory, b: SegmentCategory): number {
	return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
}

export function applyCategoryFilters(
	categories: readonly SegmentCategory[],
	counts: ReadonlyMap<string, number>,
	filters: CategoryFilters
): SegmentCategory[] {
	const query = filters.q.trim().toLowerCase();
	const matched = categories.filter((category) => {
		const count = counts.get(category.id) ?? 0;
		if (filters.has === 'segments' && count === 0) return false;
		if (filters.has === 'empty' && count > 0) return false;
		if (!query) return true;
		return [category.name, category.description ?? ''].join(' ').toLowerCase().includes(query);
	});
	if (filters.sortBy === 'count')
		return matched.sort((a, b) => (counts.get(b.id) ?? 0) - (counts.get(a.id) ?? 0) || byName(a, b));
	return matched.sort(byName);
}
