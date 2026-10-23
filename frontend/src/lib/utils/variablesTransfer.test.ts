import { describe, it, expect } from 'vitest';
import { serializeVariables, parseVariablesImport, mergeVariables, mergeVariablesDetailed, libraryImportOutcome, summarizeVariables } from './variablesTransfer';
import type { VariablesMap } from './variableDefs';

describe('serializeVariables / parseVariablesImport round-trip', () => {
	it('round-trips a text and a conditioned choice variable', () => {
		const variables: VariablesMap = {
			music: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['hip hop', 'classical'] },
			dance: {
				type: 'choice',
				mode: 'shuffle',
				pinnedIndex: null,
				options: [{ text: 'breaking', when: { var: 'music', values: ['hip hop'] } }, 'freestyle']
			},
			mood: { type: 'text', value: 'moody' }
		};

		const json = serializeVariables(variables);
		const result = parseVariablesImport(json);

		expect(result.errors).toEqual([]);
		expect(result.summary).toEqual({ variables: 3, conditions: 1 });
		expect(result.variables).toEqual(variables);
	});

	it('drops blank option rows and adjusts a pinned index that pointed past them', () => {
		const variables: VariablesMap = {
			color: { type: 'choice', mode: 'pin', pinnedIndex: 2, options: ['', 'red', '', 'blue'] }
		};

		const json = serializeVariables(variables);
		const result = parseVariablesImport(json);

		expect(result.errors).toEqual([]);
		const def = result.variables.color as { options: unknown[]; pinnedIndex: number | null };
		expect(def.options).toEqual(['red', 'blue']);
		expect(def.pinnedIndex).toBe(1);
	});
});

describe('parseVariablesImport error handling', () => {
	it('rejects invalid JSON', () => {
		const result = parseVariablesImport('{not json');
		expect(result.variables).toEqual({});
		expect(result.errors).toEqual(['Invalid JSON.']);
	});

	it('rejects a non-object payload', () => {
		const result = parseVariablesImport('[1, 2, 3]');
		expect(result.variables).toEqual({});
		expect(result.errors).toEqual(['Expected an object mapping variable names to definitions.']);
	});

	it('rejects a bad variable name', () => {
		const result = parseVariablesImport(JSON.stringify({ '2bad': { type: 'text', value: 'x' } }));
		expect(result.variables).toEqual({});
		expect(result.errors).toEqual([
			"'2bad' is not a valid variable name. Use letters, digits, and underscores, starting with a letter or underscore, up to 60 characters."
		]);
	});

	it('rejects an unrecognized definition shape', () => {
		const result = parseVariablesImport(JSON.stringify({ mood: { type: 'slider', value: 5 } }));
		expect(result.variables).toEqual({});
		expect(result.errors).toEqual(["'mood': unrecognized variable definition."]);
	});

	it('accepts a condition referencing a choice variable already on the target tab', () => {
		const existing: VariablesMap = {
			music: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['hip hop', 'classical'] }
		};
		const pasted = {
			dance: {
				type: 'choice',
				mode: 'shuffle',
				pinnedIndex: null,
				options: [{ text: 'breaking', when: { var: 'music', values: ['hip hop'] } }, 'waltz']
			}
		};

		const result = parseVariablesImport(JSON.stringify(pasted), existing);

		expect(result.errors).toEqual([]);
		expect(result.summary).toEqual({ variables: 1, conditions: 1 });
	});

	it('rejects a condition whose referenced variable is missing', () => {
		const pasted = {
			dance: {
				type: 'choice',
				mode: 'shuffle',
				pinnedIndex: null,
				options: [{ text: 'breaking', when: { var: 'musc', values: ['hip hop'] } }]
			}
		};

		const result = parseVariablesImport(JSON.stringify(pasted));

		expect(result.variables).toEqual({});
		expect(result.errors).toEqual([
			"'dance': when.var 'musc' is not a choice variable; choice variables: none defined yet."
		]);
	});

	it('rejects a cyclic condition', () => {
		const pasted = {
			a: {
				type: 'choice',
				mode: 'shuffle',
				pinnedIndex: null,
				options: [{ text: 'a1', when: { var: 'b', values: ['b1'] } }, 'a2']
			},
			b: {
				type: 'choice',
				mode: 'shuffle',
				pinnedIndex: null,
				options: [{ text: 'b1', when: { var: 'a', values: ['a1'] } }, 'b2']
			}
		};

		const result = parseVariablesImport(JSON.stringify(pasted));

		expect(result.variables.a).toBeUndefined();
		expect(result.errors.some((e) => e.includes('would make a cycle'))).toBe(true);
	});

	it('rejects a condition value that is not an exact option text', () => {
		const pasted = {
			music: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['hip hop', 'classical'] },
			dance: {
				type: 'choice',
				mode: 'shuffle',
				pinnedIndex: null,
				options: [{ text: 'breaking', when: { var: 'music', values: ['techno'] } }]
			}
		};

		const result = parseVariablesImport(JSON.stringify(pasted));

		expect(result.variables.dance).toBeUndefined();
		expect(result.variables.music).toBeDefined();
		expect(result.errors.some((e) => e.includes('are not options of $music'))).toBe(true);
	});
});

describe('mergeVariables', () => {
	const existing: VariablesMap = {
		mood: { type: 'text', value: 'existing' },
		keepme: { type: 'text', value: 'only here' }
	};
	const imported: VariablesMap = {
		mood: { type: 'text', value: 'imported' },
		fresh: { type: 'text', value: 'new' }
	};

	it('keep leaves clashing names untouched', () => {
		const merged = mergeVariables(existing, imported, 'keep');
		expect(merged.mood).toEqual({ type: 'text', value: 'existing' });
		expect(merged.fresh).toEqual({ type: 'text', value: 'new' });
		expect(merged.keepme).toEqual({ type: 'text', value: 'only here' });
	});

	it('replace overwrites clashing names', () => {
		const merged = mergeVariables(existing, imported, 'replace');
		expect(merged.mood).toEqual({ type: 'text', value: 'imported' });
		expect(merged.fresh).toEqual({ type: 'text', value: 'new' });
		expect(merged.keepme).toEqual({ type: 'text', value: 'only here' });
	});
});

describe('summarizeVariables', () => {
	it('counts variables and conditioned options', () => {
		const variables: VariablesMap = {
			music: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['hip hop', 'classical'] },
			dance: {
				type: 'choice',
				mode: 'shuffle',
				pinnedIndex: null,
				options: [{ text: 'breaking', when: { var: 'music', values: ['hip hop'] } }, 'freestyle']
			},
			mood: { type: 'text', value: 'moody' }
		};
		expect(summarizeVariables(variables)).toEqual({ variables: 3, conditions: 1 });
	});

	it('is empty for an empty map', () => {
		expect(summarizeVariables({})).toEqual({ variables: 0, conditions: 0 });
	});
});

describe('number variables in transfer', () => {
	const number = { type: 'number', mode: 'per-image', min: 0.8, max: 1.4, step: 0.1, value: 1 } as const;

	it('round-trips through serialize and parse', () => {
		const text = serializeVariables({ weight: number });
		const parsed = parseVariablesImport(text);
		expect(parsed.errors).toEqual([]);
		expect(parsed.variables.weight).toEqual(number);
		expect(parsed.summary).toEqual({ variables: 1, conditions: 0 });
	});

	it('defaults the mode to shuffle and the value to min', () => {
		const parsed = parseVariablesImport(JSON.stringify({ age: { type: 'number', min: 18, max: 80, step: 1 } }));
		expect(parsed.variables.age).toEqual({ type: 'number', mode: 'shuffle', min: 18, max: 80, step: 1, value: 18 });
	});

	it('rejects a malformed or empty range', () => {
		const missing = parseVariablesImport(JSON.stringify({ age: { type: 'number', min: 18 } }));
		expect(missing.variables).toEqual({});
		expect(missing.errors[0]).toContain('numeric min, max and step');
		const reversed = parseVariablesImport(JSON.stringify({ age: { type: 'number', min: 9, max: 1, step: 1 } }));
		expect(reversed.variables).toEqual({});
		expect(reversed.errors[0]).toContain('step above 0');
	});

	it('merges with keep and replace', () => {
		expect(mergeVariables({ age: number }, { age: { ...number, mode: 'pin' } }, 'keep').age).toEqual(number);
		expect((mergeVariables({ age: number }, { age: { ...number, mode: 'pin' } }, 'replace').age as any).mode).toBe('pin');
	});
});

describe('name collisions across case', () => {
	const text = (value: string) => ({ type: 'text', value }) as const;

	it('keeps the first of pasted names that differ only by case and notes the rest', () => {
		const parsed = parseVariablesImport(JSON.stringify({ mood: text('a'), Mood: text('b'), MOOD: text('c') }));
		expect(Object.keys(parsed.variables)).toEqual(['mood']);
		expect(parsed.notes).toHaveLength(2);
		expect(parsed.errors).toEqual([]);
	});

	it('merges a case-variant into the existing entry instead of adding a twin', () => {
		const existing: VariablesMap = { mood: text('old') };
		expect(mergeVariables(existing, { Mood: text('new') }, 'replace')).toEqual({ mood: text('new') });
		expect(mergeVariables(existing, { Mood: text('new') }, 'keep')).toEqual({ mood: text('old') });
	});

	it('never lets an imported map with case-variants create twins', () => {
		const merged = mergeVariables({}, { a: text('1'), A: text('2') } as VariablesMap, 'replace');
		expect(Object.keys(merged)).toEqual(['a']);
	});
});

describe('remapping through case-variant names', () => {
	const choice = (...options: (string | { text: string; when: { var: string; values: string[] } })[]) =>
		({ type: 'choice', mode: 'shuffle', pinnedIndex: null, options }) as const;

	it('rewrites when.var of imported definitions to the existing spelling and reports it', () => {
		const existing: VariablesMap = { mood: choice('calm', 'tense') };
		const imported: VariablesMap = {
			Mood: choice('calm', 'tense'),
			outfit: choice({ text: 'robe', when: { var: 'Mood', values: ['calm'] } }, 'coat')
		};
		const result = mergeVariablesDetailed(existing, imported, 'replace');
		expect(Object.keys(result.variables).sort()).toEqual(['mood', 'outfit']);
		const outfit = result.variables.outfit as { options: { text: string; when: { var: string } }[] };
		expect(outfit.options[0].when.var).toBe('mood');
		expect(result.remapped).toEqual([{ from: 'Mood', to: 'mood' }]);
	});

	it('rewrites a when.var that only differs by case from an existing variable', () => {
		const existing: VariablesMap = { mood: choice('calm') };
		const imported: VariablesMap = { outfit: choice({ text: 'robe', when: { var: 'MOOD', values: ['calm'] } }) };
		const result = mergeVariablesDetailed(existing, imported, 'replace');
		const outfit = result.variables.outfit as { options: { when: { var: string } }[] };
		expect(outfit.options[0].when.var).toBe('mood');
		expect(result.remapped).toEqual([{ from: 'MOOD', to: 'mood' }]);
	});

	it('does not report a remap for a variable skipped by keep', () => {
		const result = mergeVariablesDetailed({ mood: choice('a') }, { Mood: choice('b') }, 'keep');
		expect(result.remapped).toEqual([]);
		expect(result.variables.mood).toEqual(choice('a'));
	});
});

describe('libraryImportOutcome', () => {
	const text = (value: string) => ({ type: 'text', value }) as const;

	it('counts only kept variables and warns about skipped and matched names', () => {
		const outcome = libraryImportOutcome({ mood: text('old') }, { Mood: text('new'), tone: text('t'), TONE: text('u') } as VariablesMap);
		expect(outcome.importedCount).toBe(2);
		expect(outcome.successMessage).toBe('Imported 2 variables');
		expect(outcome.warningMessage).toContain('Skipped $TONE');
		expect(outcome.warningMessage).toContain('$Mood → $mood');
		expect(Object.keys(outcome.variables).sort()).toEqual(['mood', 'tone']);
	});

	it('has no warning when nothing collides', () => {
		const outcome = libraryImportOutcome({}, { a: text('1') });
		expect(outcome.successMessage).toBe('Imported 1 variable');
		expect(outcome.warningMessage).toBeNull();
	});
});
