import { describe, it, expect } from 'vitest';
import { detectConditionTrigger, isTypedCharacter } from './conditionTrigger';

describe('detectConditionTrigger', () => {
	it('matches a pair of < ending at the caret', () => {
		expect(detectConditionTrigger('a cat<<', 7)).toEqual({ start: 5, end: 7 });
		expect(detectConditionTrigger('<<', 2)).toEqual({ start: 0, end: 2 });
		expect(detectConditionTrigger('a cat<< sits', 7)).toEqual({ start: 5, end: 7 });
	});

	it('ignores a single < and a caret before the pair', () => {
		expect(detectConditionTrigger('a cat<', 6)).toBeNull();
		expect(detectConditionTrigger('<', 1)).toBeNull();
		expect(detectConditionTrigger('', 0)).toBeNull();
		expect(detectConditionTrigger('a cat<<', 5)).toBeNull();
	});

	it('ignores a run of three or more, whichever side the extra < sits on', () => {
		expect(detectConditionTrigger('a<<<', 4)).toBeNull();
		expect(detectConditionTrigger('a<<<<', 5)).toBeNull();
		expect(detectConditionTrigger('a<<<', 3)).toBeNull();
	});

	it('does not match a < pair that is not adjacent', () => {
		expect(detectConditionTrigger('a < <', 5)).toBeNull();
		expect(detectConditionTrigger('<x<', 3)).toBeNull();
	});
});

describe('isTypedCharacter', () => {
	const input = (init: { inputType?: string; data?: string | null; isComposing?: boolean }) => init as unknown as Event;

	it('accepts one typed character', () => {
		expect(isTypedCharacter(input({ inputType: 'insertText', data: '<', isComposing: false }), '<')).toBe(true);
	});

	it('rejects composition, paste, drop and multi-character insertions', () => {
		expect(isTypedCharacter(input({ inputType: 'insertText', data: '<', isComposing: true }), '<')).toBe(false);
		expect(isTypedCharacter(input({ inputType: 'insertCompositionText', data: '<', isComposing: false }), '<')).toBe(false);
		expect(isTypedCharacter(input({ inputType: 'insertFromPaste', data: '<', isComposing: false }), '<')).toBe(false);
		expect(isTypedCharacter(input({ inputType: 'insertFromDrop', data: '<', isComposing: false }), '<')).toBe(false);
		expect(isTypedCharacter(input({ inputType: 'insertText', data: '<<', isComposing: false }), '<')).toBe(false);
		expect(isTypedCharacter(input({ inputType: 'insertText', data: null, isComposing: false }), '<')).toBe(false);
	});

	it('rejects an event that carries no input details', () => {
		expect(isTypedCharacter(input({}), '<')).toBe(false);
	});
});
