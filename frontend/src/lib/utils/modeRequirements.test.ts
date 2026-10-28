import { describe, it, expect } from 'vitest';
import { modeRequirementIssue, modeUnavailableReason } from './modeRequirements';

describe('modeRequirementIssue', () => {
	it('is silent when nothing is missing or the verdict is unknown', () => {
		expect(modeRequirementIssue([])).toBeUndefined();
		expect(modeRequirementIssue(null)).toBeUndefined();
		expect(modeRequirementIssue(undefined)).toBeUndefined();
	});
	it('words one and several missing files', () => {
		expect(modeRequirementIssue(['a'])).toBe('A required file is missing.');
		expect(modeRequirementIssue(['a', 'b'])).toBe('2 required files are missing.');
	});
});

describe('modeUnavailableReason', () => {
	const modes = [
		{ name: 'txt2vid', requirements_missing: [] },
		{ name: 'colorize', requirements_missing: ['abc123'] }
	];
	it('blocks only the selected mode that has a missing file', () => {
		expect(modeUnavailableReason(modes, 'colorize')).toBe('A required file is missing.');
		expect(modeUnavailableReason(modes, 'txt2vid')).toBeUndefined();
	});
	it('does not block without a selection or an unknown mode', () => {
		expect(modeUnavailableReason(modes, null)).toBeUndefined();
		expect(modeUnavailableReason(modes, 'other')).toBeUndefined();
		expect(modeUnavailableReason(undefined, 'colorize')).toBeUndefined();
	});
});
