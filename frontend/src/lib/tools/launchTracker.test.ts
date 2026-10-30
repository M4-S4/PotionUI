import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
vi.mock('$lib/stores/toast', () => ({ toasts: { show: vi.fn() } }));
vi.mock('$lib/media-tools/runs', () => ({ mediaToolRuns: { start: vi.fn() } }));

import { executeMediaTool } from '$lib/media-tools/execute';
import { trackLaunch } from './launchTracker';

type Outcome = { status: 'completed' | 'failed' | 'cancelled' };

function fakeRuns(outcome: Promise<Outcome>) {
	return { start: vi.fn().mockResolvedValue({ runId: 'r1', done: outcome }) } as never;
}

async function start(outcome: Promise<Outcome>) {
	return executeMediaTool({
		label: 'Trim',
		steps: [{ tool: 'trim', params: {} }],
		sources: [{ id: 's', kind: 'video', url: '/a', filename: 'a.mp4' }],
		refs: [],
		mode: 'new',
		runs: fakeRuns(outcome)
	});
}

const flush = async () => {
	for (let i = 0; i < 4; i += 1) await Promise.resolve();
};

describe('trackLaunch', () => {
	let abandon: ReturnType<typeof vi.fn<() => void>>;
	beforeEach(() => {
		abandon = vi.fn<() => void>();
	});

	it('abandons when the editor closes without a run having started', () => {
		const tracker = trackLaunch(abandon);

		tracker.closed();

		expect(abandon).toHaveBeenCalledTimes(1);
	});

	it('does not abandon on close when a run started, and waits for its outcome', async () => {
		const tracker = trackLaunch(abandon);
		await start(Promise.resolve({ status: 'completed' }));
		tracker.closed();
		await flush();

		expect(abandon).not.toHaveBeenCalled();
	});

	it('abandons when the run failed', async () => {
		const tracker = trackLaunch(abandon);
		await start(Promise.resolve({ status: 'failed' }));
		tracker.closed();
		await flush();

		expect(abandon).toHaveBeenCalledTimes(1);
	});

	it('abandons when the run was cancelled', async () => {
		const tracker = trackLaunch(abandon);
		await start(Promise.resolve({ status: 'cancelled' }));
		await flush();

		expect(abandon).toHaveBeenCalledTimes(1);
	});

	it('does not abandon once a result was delivered', async () => {
		const tracker = trackLaunch(abandon);
		tracker.delivered();
		tracker.closed();

		expect(abandon).not.toHaveBeenCalled();
	});

	it('does not abandon a completed run', async () => {
		trackLaunch(abandon);
		await start(Promise.resolve({ status: 'completed' }));
		await flush();

		expect(abandon).not.toHaveBeenCalled();
	});
});
