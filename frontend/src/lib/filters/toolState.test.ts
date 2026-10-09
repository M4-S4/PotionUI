import { describe, expect, it } from 'vitest';
import {
	canApply,
	clampIntensity,
	historyLabel,
	initialToolState,
	selectActive,
	withCompare,
	withIntensity
} from './toolState';

const EMBER = { id: 'ember' };

describe('intensity', () => {
	it('clamps to 0..100 and rounds', () => {
		expect(clampIntensity(-5)).toBe(0);
		expect(clampIntensity(140)).toBe(100);
		expect(clampIntensity(69.6)).toBe(70);
		expect(clampIntensity(Number.NaN)).toBe(100);
	});

	it('selecting takes the filter default and clears compare', () => {
		const held = withCompare(selectActive(initialToolState<typeof EMBER>(), EMBER, 100), true);
		const next = selectActive(held, EMBER, 55);
		expect(next).toEqual({ active: EMBER, intensity: 55, compare: false });
	});

	it('selecting nothing returns the initial state', () => {
		const state = withIntensity(selectActive(initialToolState<typeof EMBER>(), EMBER, 100), 40);
		expect(selectActive(state, null)).toEqual(initialToolState());
	});

	it('withIntensity clamps', () => {
		const state = selectActive(initialToolState<typeof EMBER>(), EMBER, 100);
		expect(withIntensity(state, 250).intensity).toBe(100);
		expect(withIntensity(state, -1).intensity).toBe(0);
	});
});

describe('compare', () => {
	it('holds compare only with a filter selected', () => {
		const idle = initialToolState<typeof EMBER>();
		expect(withCompare(idle, true).compare).toBe(false);

		const state = withIntensity(selectActive(idle, EMBER, 100), 70);
		const held = withCompare(state, true);
		expect(held.compare).toBe(true);
		expect(held.intensity).toBe(70);
		expect(withCompare(held, false).compare).toBe(false);
	});
});

describe('apply and history label', () => {
	it('can only apply a selected filter above zero', () => {
		const idle = initialToolState<typeof EMBER>();
		expect(canApply(idle)).toBe(false);
		const state = selectActive(idle, EMBER, 70);
		expect(canApply(state)).toBe(true);
		expect(canApply(withIntensity(state, 0))).toBe(false);
	});

	it('names the undo step after the filter and intensity', () => {
		expect(historyLabel('Ember', 70)).toBe('Filter: Ember 70%');
		expect(historyLabel('Ember', 100.2)).toBe('Filter: Ember 100%');
	});
});
