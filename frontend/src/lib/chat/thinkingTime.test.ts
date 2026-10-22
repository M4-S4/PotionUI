import { describe, expect, it } from 'vitest';
import { formatThinkingElapsed } from './thinkingTime';

describe('formatThinkingElapsed', () => {
	it('shows a sub-second wait without a misleading zero', () => {
		expect(formatThinkingElapsed(0)).toBe('<1s');
		expect(formatThinkingElapsed(999)).toBe('<1s');
	});

	it('rounds to whole seconds under a minute', () => {
		expect(formatThinkingElapsed(1000)).toBe('1s');
		expect(formatThinkingElapsed(1499)).toBe('1s');
		expect(formatThinkingElapsed(1500)).toBe('2s');
		expect(formatThinkingElapsed(59000)).toBe('59s');
	});

	it('switches to minutes with a padded seconds part', () => {
		expect(formatThinkingElapsed(60000)).toBe('1m 00s');
		expect(formatThinkingElapsed(65000)).toBe('1m 05s');
		expect(formatThinkingElapsed(605000)).toBe('10m 05s');
	});
});
