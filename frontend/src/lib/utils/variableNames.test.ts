import { describe, it, expect } from 'vitest';
import { dedupeVariableNames, findNameCollision, DUPLICATE_NAME_ERROR, INVALID_NAME_ERROR, usableNamedRows, variableNameErrors } from './variableNames';

const rows = (...names: string[]) => names.map((name, i) => ({ key: `k${i}`, name }));

describe('variableNameErrors', () => {
	it('flags identical names on every holder', () => {
		const errors = variableNameErrors(rows('mood', 'mood'));
		expect([...errors.values()]).toEqual([DUPLICATE_NAME_ERROR, DUPLICATE_NAME_ERROR]);
	});

	it('flags names that differ only by case or surrounding whitespace', () => {
		expect([...variableNameErrors(rows('mood', 'Mood')).values()]).toEqual([DUPLICATE_NAME_ERROR, DUPLICATE_NAME_ERROR]);
		expect([...variableNameErrors(rows('mood', 'mood ')).values()]).toEqual([DUPLICATE_NAME_ERROR, DUPLICATE_NAME_ERROR]);
	});

	it('leaves distinct and empty names alone, and flags invalid ones', () => {
		const errors = variableNameErrors(rows('mood', 'mood2', '', '2x'));
		expect([...errors.values()]).toEqual([null, null, null, INVALID_NAME_ERROR]);
	});
});

describe('usableNamedRows', () => {
	it('keeps the first of each case-insensitive name and drops invalid or empty ones', () => {
		const kept = usableNamedRows(rows('mood', 'Mood', ' mood', 'ok', '', '2x'));
		expect(kept.map((r) => r.name)).toEqual(['mood', 'ok']);
	});
});

describe('findNameCollision / dedupeVariableNames', () => {
	it('finds the existing spelling ignoring case and whitespace', () => {
		expect(findNameCollision({ mood: 1 }, 'MOOD ')).toBe('mood');
		expect(findNameCollision({ mood: 1 }, 'mood2')).toBeNull();
		expect(findNameCollision({ mood: 1 }, '')).toBeNull();
	});

	it('keeps the first spelling and reports the dropped ones', () => {
		const result = dedupeVariableNames({ mood: 1, Mood: 2, other: 3 });
		expect(result.variables).toEqual({ mood: 1, other: 3 });
		expect(result.dropped).toEqual(['Mood']);
	});
});
