import { createFilterCodec, type FilterFieldDescriptor } from '$lib/components/library/filterCodec';
import type { SortOption } from '$lib/components/library/librarySection';
import type { RecipeSummary } from '$lib/services/api/recipes';

export type RecipeSortBy = 'name' | 'category';

export interface RecipeFilters {
	q: string;
	source: string;
	engine: string;
	sortBy: RecipeSortBy;
}

export const DEFAULT_RECIPE_FILTERS: RecipeFilters = { q: '', source: '', engine: '', sortBy: 'name' };

export const RECIPE_SORT_OPTIONS: readonly SortOption<RecipeSortBy>[] = [
	{ value: 'name', label: 'Name' },
	{ value: 'category', label: 'Category' }
];

const FIELDS: readonly FilterFieldDescriptor<RecipeFilters>[] = [
	{
		kind: 'text',
		key: 'source',
		param: 'source',
		label: 'Source',
		default: '',
		chipLabel: (value) => `source: ${value}`
	},
	{
		kind: 'text',
		key: 'engine',
		param: 'engine',
		label: 'Engine',
		default: '',
		chipLabel: (value) => `engine: ${value}`
	}
];

export const recipeFilters = createFilterCodec<RecipeFilters>({
	defaults: DEFAULT_RECIPE_FILTERS,
	fields: FIELDS,
	sortValues: ['name', 'category']
});

export function recipeSourceVocabulary(recipes: readonly RecipeSummary[]): string[] {
	return [...new Set(recipes.map((recipe) => recipe.source).filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

export function recipeEngineVocabulary(recipes: readonly RecipeSummary[]): string[] {
	return [...new Set(recipes.map((recipe) => recipe.engine).filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

export function applyRecipeFilters(recipes: readonly RecipeSummary[], filters: RecipeFilters): RecipeSummary[] {
	const query = filters.q.trim().toLowerCase();
	const rows = recipes.filter((recipe) => {
		if (filters.source && recipe.source !== filters.source) return false;
		if (filters.engine && recipe.engine !== filters.engine) return false;
		if (!query) return true;
		const haystack = [recipe.name, recipe.id, recipe.description].filter(Boolean).join(' ').toLowerCase();
		return haystack.includes(query);
	});
	if (filters.sortBy === 'category') {
		return rows.sort((a, b) => a.category.localeCompare(b.category) || a.name.localeCompare(b.name));
	}
	return rows.sort((a, b) => a.name.localeCompare(b.name));
}
