import { describe, it, expect } from 'vitest';
import { numberBoxIssue, parseNumberBox } from './numberBox';
import { normalizeHex } from './colorValue';
import { anchorLabel, isAnchorValue } from './anchorValues';

describe('parseNumberBox', () => {
	it('reads whole numbers, decimals and negatives', () => {
		expect(parseNumberBox('12', false)).toBe(12);
		expect(parseNumberBox(' 1.5 ', false)).toBe(1.5);
		expect(parseNumberBox('-3', false)).toBe(-3);
		expect(parseNumberBox('7.', false)).toBe(7);
	});

	it('rounds when the box is for integers', () => {
		expect(parseNumberBox('2.5', true)).toBe(3);
		expect(parseNumberBox('2.4', true)).toBe(2);
	});

	it('reads an empty box as null, meaning unset', () => {
		expect(parseNumberBox('', false)).toBeNull();
		expect(parseNumberBox('   ', false)).toBeNull();
	});

	it('refuses half-typed and non-numeric text', () => {
		for (const text of ['abc', '-', '.', '-.', '1e5', '1,5', '--1', '1.2.3']) {
			expect(parseNumberBox(text, false)).toBeUndefined();
		}
	});
});

describe('numberBoxIssue', () => {
	it('is quiet for a value inside its bounds and for an empty box', () => {
		expect(numberBoxIssue(5, '5', 1, 10)).toBeNull();
		expect(numberBoxIssue(null, '', 1, 10)).toBeNull();
	});

	it('names the bound that is broken', () => {
		expect(numberBoxIssue(0, '0', 1, 10)).toBe('At least 1');
		expect(numberBoxIssue(11, '11', 1, 10)).toBe('At most 10');
	});

	it('asks for a number when the text is not one', () => {
		expect(numberBoxIssue(5, 'abc', 1, 10)).toBe('Enter a number');
	});

	it('ignores bounds that are not numbers', () => {
		expect(numberBoxIssue(5, '5', undefined, null)).toBeNull();
	});
});

describe('normalizeHex', () => {
	it('upper-cases, adds the hash and expands three digits', () => {
		expect(normalizeHex('#00b140')).toBe('#00B140');
		expect(normalizeHex('00b140')).toBe('#00B140');
		expect(normalizeHex('#0f0')).toBe('#00FF00');
		expect(normalizeHex('  #abc  ')).toBe('#AABBCC');
	});

	it('refuses anything else', () => {
		for (const value of ['', '#12', '#12345', '#1234567', '#12zz56', 'green', 12, null, undefined]) {
			expect(normalizeHex(value)).toBeNull();
		}
	});
});

describe('anchor values', () => {
	it('knows the nine anchors and writes a label for one', () => {
		expect(isAnchorValue('bottom_right')).toBe(true);
		expect(isAnchorValue('middle')).toBe(false);
		expect(anchorLabel('top_left')).toBe('Top left');
		expect(anchorLabel('center')).toBe('Center');
	});
});
