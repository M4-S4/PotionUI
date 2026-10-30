import { describe, expect, it } from 'vitest';
import { NO_PROMPT_COPY, promptModeCopy, promptTitle } from './promptModeCopy';

const vars = {
	prompt: {
		modes: {
			text: { label: 'Placement and style', placeholder: 'in white capitals at the top' },
			blank: { label: '   ', placeholder: 42 }
		}
	}
};

describe('promptModeCopy', () => {
	it('reads the label and placeholder declared for the mode', () => {
		expect(promptModeCopy(vars, 'text', true)).toEqual({
			label: 'Placement and style',
			placeholder: 'in white capitals at the top',
			optional: true
		});
	});

	it('falls back to nothing for an undeclared mode, bad values or missing vars', () => {
		expect(promptModeCopy(vars, 'other', false)).toEqual(NO_PROMPT_COPY);
		expect(promptModeCopy(vars, 'blank', false)).toEqual(NO_PROMPT_COPY);
		expect(promptModeCopy(null, 'text', false)).toEqual(NO_PROMPT_COPY);
		expect(promptModeCopy({ prompt: 'x' }, 'text', false)).toEqual(NO_PROMPT_COPY);
	});
});

describe('promptTitle', () => {
	it('uses the label and marks an optional prompt', () => {
		expect(promptTitle({ label: 'Scene', placeholder: null, optional: true })).toBe('Scene (optional)');
		expect(promptTitle({ label: 'Scene', placeholder: null, optional: false })).toBe('Scene');
	});

	it('keeps Prompt when the mode declares no label', () => {
		expect(promptTitle(NO_PROMPT_COPY)).toBe('Prompt');
		expect(promptTitle({ ...NO_PROMPT_COPY, optional: true })).toBe('Prompt (optional)');
	});
});
