import { describe, it, expect } from 'vitest';
import { buildVariableChipTooltips } from './variableSnapshot';
import type { VariablesMap, VariableRoll } from './variableDefs';

describe('buildVariableChipTooltips', () => {
	it('returns an empty map for undefined variables', () => {
		expect(buildVariableChipTooltips(undefined)).toEqual({});
	});

	it('describes a choice variable in plain language with the last roll', () => {
		const vars: VariablesMap = {
			mood: { type: 'choice', options: ['noir', 'sunlit'], mode: 'shuffle', pinnedIndex: null }
		};
		const rolls: Record<string, VariableRoll> = {
			mood: { optionIndex: 1, value: 'sunlit', rolledAt: 1 }
		};
		expect(buildVariableChipTooltips(vars, rolls)).toEqual({
			mood: 'one of noir, sunlit — shuffles each generation; last roll: sunlit'
		});
	});

	it('marks a defined-but-empty choice as known (mid-authoring)', () => {
		const vars: VariablesMap = {
			mood: { type: 'choice', options: ['', ''], mode: 'shuffle', pinnedIndex: null }
		};
		const tips = buildVariableChipTooltips(vars);
		expect('mood' in tips).toBe(true);
	});

	it('describes a pinned choice by naming the pinned option', () => {
		const vars: VariablesMap = {
			mood: { type: 'choice', options: ['noir', 'sunlit'], mode: 'pin', pinnedIndex: 1 }
		};
		expect(buildVariableChipTooltips(vars).mood).toBe('one of noir, sunlit — pinned to sunlit');
	});

	it('describes a text variable by its value', () => {
		const vars: VariablesMap = { subject: { type: 'text', value: 'a fox' } };
		expect(buildVariableChipTooltips(vars).subject).toBe('a fox');
	});
});


describe('number variable chip tooltips', () => {
	const number = { type: 'number', mode: 'shuffle', min: 18, max: 80, step: 1, value: 47 } as const;

	it('writes the tooltip for a chat chip', () => {
		expect(buildVariableChipTooltips({ age: { ...number, mode: 'pin', value: 47 } }).age).toBe(
			'a number from 18 to 80 in steps of 1 — pinned to 47'
		);
		expect(buildVariableChipTooltips({ age: { ...number, mode: 'per-image' } }).age).toContain('re-rolls per image');
	});
});
