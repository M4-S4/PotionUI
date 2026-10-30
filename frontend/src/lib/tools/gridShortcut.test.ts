import { describe, expect, it } from 'vitest';
import { gridShortcutTarget } from './gridShortcut';

describe('gridShortcutTarget', () => {
	it('is the one selected item, whatever has focus', () => {
		expect(gridShortcutTarget(['a'], 'b')).toBe('a');
	});

	it('is the focused card when nothing is selected', () => {
		expect(gridShortcutTarget([], 'b')).toBe('b');
	});

	it('is nothing when nothing is selected and no card has focus', () => {
		expect(gridShortcutTarget([], null)).toBeNull();
	});

	it('is nothing when several items are selected', () => {
		expect(gridShortcutTarget(['a', 'b'], 'a')).toBeNull();
	});
});
