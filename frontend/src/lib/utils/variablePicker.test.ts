import { describe, it, expect } from 'vitest';
import {
	creatableVariableName,
	normalizePickerQuery,
	variableModeLabel,
	variablePickerRows,
	variableValuePreview
} from './variablePicker';
import { normalizeVariableDef, type VariablesMap } from './variableDefs';

const variables: VariablesMap = {
	mood: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['calm', 'serene', 'tense', 'wild', 'quiet'] },
	hair: { type: 'choice', mode: 'pin', pinnedIndex: 0, options: ['auburn', 'grey'] },
	age: { type: 'number', mode: 'per-image', min: 18, max: 80, step: 1, value: 18 },
	name: { type: 'text', value: 'Mara' },
	moody: { type: 'text', value: 'x' }
};

describe('variablePickerRows', () => {
	it('returns every variable with type, mode and preview when the query is empty', () => {
		const rows = variablePickerRows(variables, '');
		expect(rows.map((r) => r.name)).toEqual(['mood', 'hair', 'age', 'name', 'moody']);
		const mood = rows.find((r) => r.name === 'mood')!;
		expect(mood.typeLabel).toBe('Choice');
		expect(mood.mode).toBe('shuffle');
		expect(mood.preview).toBe('calm · serene · tense +2');
		expect(rows.find((r) => r.name === 'age')).toMatchObject({ typeLabel: 'Number', mode: 'per image', preview: '18–80' });
		expect(rows.find((r) => r.name === 'name')).toMatchObject({ typeLabel: 'Text', mode: '', preview: 'Mara' });
	});

	it('filters by name case-insensitively, prefix matches first', () => {
		const rows = variablePickerRows(variables, 'MOO');
		expect(rows.map((r) => r.name)).toEqual(['mood', 'moody']);
	});

	it('matches value text and ranks it after name matches', () => {
		expect(variablePickerRows(variables, 'serene').map((r) => r.name)).toEqual(['mood']);
		expect(variablePickerRows(variables, 'ma').map((r) => r.name)).toEqual(['name']);
		expect(variablePickerRows(variables, 'a').map((r) => r.name)).toEqual(['age', 'hair', 'name', 'mood']);
	});

	it('ignores a leading $ or ${ in the query', () => {
		expect(variablePickerRows(variables, '$age').map((r) => r.name)).toEqual(['age']);
		expect(variablePickerRows(variables, '${hair}').map((r) => r.name)).toEqual(['hair']);
		expect(normalizePickerQuery('  $x ')).toBe('x');
	});

	it('restricts to allowed types', () => {
		expect(variablePickerRows(variables, '', { allowedTypes: ['number'] }).map((r) => r.name)).toEqual(['age']);
		expect(variablePickerRows(variables, 'a', { allowedTypes: ['text', 'number'] }).map((r) => r.name)).toEqual(['age', 'name']);
	});

	it('carries disabled reasons without dropping the row', () => {
		const rows = variablePickerRows(variables, '', { disabledReasons: { age: 'number variable' } });
		expect(rows.find((r) => r.name === 'age')!.disabledReason).toBe('number variable');
		expect(rows.find((r) => r.name === 'mood')!.disabledReason).toBeNull();
	});
});

describe('labels', () => {
	it('names the mode', () => {
		expect(variableModeLabel(normalizeVariableDef(variables.hair))).toBe('pinned');
		expect(variableModeLabel(normalizeVariableDef(variables.name))).toBe('');
	});

	it('previews empty choices as empty', () => {
		expect(variableValuePreview({ type: 'choice', options: ['', ' '], mode: 'shuffle', pinnedIndex: null })).toBe('');
	});
});

describe('creatableVariableName', () => {
	it('offers a valid unused name', () => {
		expect(creatableVariableName('hat', variables)).toBe('hat');
		expect(creatableVariableName('$hat', variables)).toBe('hat');
	});

	it('refuses existing, invalid and empty names', () => {
		expect(creatableVariableName('mood', variables)).toBeNull();
		expect(creatableVariableName('my hat', variables)).toBeNull();
		expect(creatableVariableName('2x', variables)).toBeNull();
		expect(creatableVariableName('', variables)).toBeNull();
	});
});
