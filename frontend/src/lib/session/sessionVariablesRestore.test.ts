import { describe, it, expect, vi } from 'vitest';
import { decideSessionVariables, finishDeferredSessionVariables } from './sessionVariablesRestore';

const text = (value: string) => ({ type: 'text' as const, value });

describe('decideSessionVariables', () => {
	it('keeps the draft variables when the draft already has some', () => {
		expect(decideSessionVariables(true, { mood: text('noir') })).toBe('keep');
	});

	it('replaces with the session variables when the settled draft is empty', () => {
		expect(decideSessionVariables(true, {})).toBe('replace');
		expect(decideSessionVariables(true, undefined)).toBe('replace');
	});

	it('defers when boot has not finished, whatever the tab holds', () => {
		expect(decideSessionVariables(false, {})).toBe('defer');
		expect(decideSessionVariables(false, { mood: text('noir') })).toBe('defer');
	});
});

describe('finishDeferredSessionVariables', () => {
	function deps(ready: boolean, current: Record<string, ReturnType<typeof text>> | undefined) {
		return {
			settled: async () => ready,
			currentVariables: () => current,
			apply: vi.fn(),
			replace: vi.fn()
		};
	}

	it('does nothing when the server draft hydrated variables', async () => {
		const d = deps(true, { fromServer: text('1') });
		expect(await finishDeferredSessionVariables(d, { style: text('a') })).toBe('keep');
		expect(d.apply).not.toHaveBeenCalled();
		expect(d.replace).not.toHaveBeenCalled();
	});

	it('applies and replaces once when the server has no draft', async () => {
		const d = deps(true, {});
		expect(await finishDeferredSessionVariables(d, { style: text('a') })).toBe('replace');
		expect(d.apply).toHaveBeenCalledWith({ style: text('a') });
		expect(d.replace).toHaveBeenCalledWith({ style: text('a') });
	});

	it('does nothing when the sync is stopped before boot completes', async () => {
		const d = deps(false, {});
		expect(await finishDeferredSessionVariables(d, { style: text('a') })).toBe('defer');
		expect(d.apply).not.toHaveBeenCalled();
		expect(d.replace).not.toHaveBeenCalled();
	});
});
