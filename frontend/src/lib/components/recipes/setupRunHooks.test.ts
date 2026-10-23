import { describe, it, expect, vi } from 'vitest';
import type { SetupRun } from '$lib/services/api/setup';
import { createSetupRunHooks, SETUP_POLL_RETRY_NOTICE } from './setupRunHooks';

function setup() {
	const target = { setRun: vi.fn(), setFetchError: vi.fn(), remember: vi.fn(), forget: vi.fn() };
	return { target, hooks: createSetupRunHooks(target) };
}

describe('createSetupRunHooks', () => {
	it('mirrors a run, clears the fetch error and remembers its id', () => {
		const { target, hooks } = setup();
		const run = { id: 'run-1' } as SetupRun;

		hooks.onRun(run);

		expect(target.setRun).toHaveBeenCalledWith(run);
		expect(target.setFetchError).toHaveBeenCalledWith('');
		expect(target.remember).toHaveBeenCalledWith('run-1');
		expect(target.forget).not.toHaveBeenCalled();
	});

	it('forgets the stored id when the run is cleared', () => {
		const { target, hooks } = setup();

		hooks.onRun(null);

		expect(target.setRun).toHaveBeenCalledWith(null);
		expect(target.forget).toHaveBeenCalledTimes(1);
		expect(target.remember).not.toHaveBeenCalled();
	});

	it('drops the run quietly on a 404', () => {
		const { target, hooks } = setup();

		expect(hooks.onPollError({ response: { status: 404 } })).toBe('drop');
		expect(target.setFetchError).not.toHaveBeenCalled();
	});

	it('shows the retry notice and retries on any other error', () => {
		const { target, hooks } = setup();

		expect(hooks.onPollError({ response: { status: 500 } })).toBe('retry');
		expect(hooks.onPollError(new Error('network'))).toBe('retry');
		expect(target.setFetchError).toHaveBeenCalledWith(SETUP_POLL_RETRY_NOTICE);
	});
});
