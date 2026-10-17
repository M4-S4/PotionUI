import { describe, expect, it } from 'vitest';
import { toolsStepLabel } from './traceLabels';

describe('toolsStepLabel', () => {
	it('says Tools off when nothing is offered', () => {
		expect(toolsStepLabel({ offered: [], withheld: { a: 'off_by_toggle' } })).toBe('Tools off');
		expect(toolsStepLabel(undefined)).toBe('Tools off');
	});

	it('does not alarm about tools that only lack page context', () => {
		expect(
			toolsStepLabel({ offered: ['a', 'b'], withheld: { c: 'unavailable', d: 'unavailable' } })
		).toBe('Tools: 2 ready');
	});

	it('counts the tools the user or an admin switched off', () => {
		expect(
			toolsStepLabel({
				offered: ['a'],
				withheld: { b: 'opted_out', c: 'disabled_by_admin', d: 'unavailable' }
			})
		).toBe('Tools: 1 ready, 2 off');
	});
});
