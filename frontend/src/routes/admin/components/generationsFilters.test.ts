import { describe, it, expect } from 'vitest';
import {
	DEFAULT_GENERATIONS_FILTERS,
	generationsFilterChips,
	generationsSortParams,
	type GenerationsFilters,
	generationsFilters
} from './generationsFilters';

describe('generationsFilters.fromSearchParams / generationsFilters.toSearchParams', () => {
	it('round-trips a fully-set filter state through the URL', () => {
		const filters: GenerationsFilters = {
			q: 'portrait',
			status: 'completed',
			category: '',
			userId: 'user-1',
			createdFrom: '2026-09-01',
			createdTo: '2026-09-20',
			sortBy: 'created_asc'
		};
		const params = generationsFilters.toSearchParams(filters);
		expect(params.get('q')).toBe('portrait');
		expect(params.get('status')).toBe('completed');
		expect(params.get('category')).toBeNull();
		expect(params.get('user')).toBe('user-1');
		expect(params.get('from')).toBe('2026-09-01');
		expect(params.get('to')).toBe('2026-09-20');
		expect(params.get('sort_by')).toBe('created_asc');
		expect(generationsFilters.fromSearchParams(params)).toEqual(filters);
	});

	it('round-trips the category filter through the URL', () => {
		const filters: GenerationsFilters = {
			q: '',
			status: 'failed',
			category: 'cuda_oom',
			userId: '',
			createdFrom: '',
			createdTo: '',
			sortBy: 'created_desc'
		};
		const params = generationsFilters.toSearchParams(filters);
		expect(params.get('category')).toBe('cuda_oom');
		expect(generationsFilters.fromSearchParams(params)).toEqual(filters);
	});

	it('omits defaults from the URL so a clean list has no query string', () => {
		expect(generationsFilters.toSearchParams(DEFAULT_GENERATIONS_FILTERS).toString()).toBe('');
	});

	it('falls back to the defaults for an unknown status, category or sort value', () => {
		const params = new URLSearchParams({ status: 'archived', category: 'not_a_real_category', sort_by: 'rating' });
		const filters = generationsFilters.fromSearchParams(params);
		expect(filters.status).toBe('');
		expect(filters.category).toBe('');
		expect(filters.sortBy).toBe('created_desc');
	});
});

describe('generations filter chips', () => {
	it('emits one chip per active field with a friendly status label and clears them individually', () => {
		const filters: GenerationsFilters = {
			q: 'x',
			status: 'failed',
			category: 'cuda_oom',
			userId: 'user-1',
			createdFrom: '2026-09-01',
			createdTo: '2026-09-20',
			sortBy: 'created_desc'
		};
		const chips = generationsFilterChips(filters, (userId) => `user:${userId}`);
		expect(chips).toEqual([
			{ key: 'status', label: 'Failed' },
			{ key: 'category', label: 'GPU out of memory' },
			{ key: 'userId', label: 'user:user-1' },
			{ key: 'createdFrom', label: 'From 2026-09-01' },
			{ key: 'createdTo', label: 'To 2026-09-20' }
		]);
		expect(generationsFilters.clearChip(filters, 'status').status).toBe('');
		expect(generationsFilters.clearChip(filters, 'category').category).toBe('');
		expect(generationsFilters.clearChip(filters, 'userId').userId).toBe('');
		expect(generationsFilters.clearChip(filters, 'nothing')).toBe(filters);
	});

	it('counts every active field and keeps query and sort when clearing all', () => {
		const filters: GenerationsFilters = {
			q: 'keep me',
			status: 'running',
			category: 'disk_full',
			userId: 'user-2',
			createdFrom: '2026-09-01',
			createdTo: '',
			sortBy: 'created_asc'
		};
		expect(generationsFilters.activeCount(filters)).toBe(4);
		expect(generationsFilters.clearAll(filters)).toEqual({
			q: 'keep me',
			status: '',
			category: '',
			userId: '',
			createdFrom: '',
			createdTo: '',
			sortBy: 'created_asc'
		});
		expect(generationsFilters.activeCount(DEFAULT_GENERATIONS_FILTERS)).toBe(0);
	});
});

describe('generationsSortParams', () => {
	it('maps the sort option to the backend column and direction', () => {
		expect(generationsSortParams('created_desc')).toEqual({ sortBy: 'created_at', sortDir: 'desc' });
		expect(generationsSortParams('created_asc')).toEqual({ sortBy: 'created_at', sortDir: 'asc' });
	});
});
