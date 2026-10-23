import { describe, it, expect } from 'vitest';
import { contextToolsAvailable } from './contextTools';

describe('contextToolsAvailable', () => {
	it('is off until a preset is chosen', () => {
		expect(contextToolsAvailable({ selectedPreset: '' })).toBe(false);
		expect(contextToolsAvailable({ selectedPreset: null })).toBe(false);
		expect(contextToolsAvailable({ selectedPreset: undefined as never })).toBe(false);
		expect(contextToolsAvailable(null)).toBe(false);
		expect(contextToolsAvailable(undefined)).toBe(false);
	});

	it('is on for a chosen preset, whatever the mode', () => {
		expect(contextToolsAvailable({ selectedPreset: 'krea2' })).toBe(true);
	});
});
