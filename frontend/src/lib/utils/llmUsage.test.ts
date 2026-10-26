import { describe, expect, it } from 'vitest';
import { formatCost, formatTokens, traceCost, traceSummary, usageRows } from './llmUsage';

describe('formatCost', () => {
	it('keeps enough digits to read a fraction of a cent', () => {
		expect(formatCost(0.0031)).toBe('$0.0031');
		expect(formatCost(0.052)).toBe('$0.052');
		expect(formatCost(0.00002)).toBe('$0.000020');
	});

	it('shows two decimals from a dollar up and a bare zero for free calls', () => {
		expect(formatCost(1.5)).toBe('$1.50');
		expect(formatCost(12)).toBe('$12.00');
		expect(formatCost(0)).toBe('$0');
	});

	it('shows a dash for an unknown cost', () => {
		expect(formatCost(null)).toBe('—');
		expect(formatCost(undefined)).toBe('—');
		expect(formatCost(Number.NaN)).toBe('—');
	});
});

describe('formatTokens', () => {
	it('groups thousands and marks an unknown count', () => {
		expect(formatTokens(5210)).toBe('5,210');
		expect(formatTokens(0)).toBe('0');
		expect(formatTokens(null)).toBe('?');
		expect(formatTokens(undefined)).toBe('?');
	});
});

describe('traceCost', () => {
	it('prefers the trace cost and falls back to the usage cost', () => {
		expect(traceCost({ cost: 0.5, usage: { cost: 0.7 } })).toBe(0.5);
		expect(traceCost({ usage: { cost: 0.7 } })).toBe(0.7);
		expect(traceCost({})).toBeNull();
	});
});

describe('traceSummary', () => {
	it('shows cached tokens, reasoning tokens and cost beside the token flow', () => {
		expect(
			traceSummary({
				prompt_tokens: 1,
				completion_tokens: 1,
				usage: { input_tokens: 5210, output_tokens: 312, cached_input_tokens: 4800, reasoning_tokens: 180 },
				cost: 0.0031
			})
		).toBe('5,210→312 tok · 4,800 cached · 180 reasoning · $0.0031');
	});

	it('leaves out what was not reported', () => {
		expect(traceSummary({ usage: { input_tokens: 10, output_tokens: 2, cached_input_tokens: 0 } })).toBe('10→2 tok');
	});

	it('falls back to the prompt and completion tokens of an older trace', () => {
		expect(traceSummary({ prompt_tokens: 900, completion_tokens: 40 })).toBe('900→40 tok');
		expect(traceSummary({ prompt_tokens: null, completion_tokens: null })).toBe('?→? tok');
	});
});

describe('usageRows', () => {
	it('lists every reported figure with its label, cost last', () => {
		expect(
			usageRows({
				usage: {
					input_tokens: 5210,
					output_tokens: 312,
					reasoning_tokens: 180,
					cached_input_tokens: 4800,
					cache_write_tokens: 0,
					cost: 0.0031
				}
			}).map((row) => [row.label, row.value])
		).toEqual([
			['Input', '5,210'],
			['Output', '312'],
			['Cached input', '4,800'],
			['Cache write', '0'],
			['Reasoning', '180'],
			['Cost', '$0.0031']
		]);
	});

	it('is empty for a trace that reports nothing', () => {
		expect(usageRows({})).toEqual([]);
	});

	it('uses the top-level cost when the usage has none', () => {
		const rows = usageRows({ usage: { input_tokens: 1 }, cost: 0.01 });
		expect(rows.at(-1)).toEqual({ key: 'cost', label: 'Cost', value: '$0.010' });
	});
});
