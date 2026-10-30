import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
	MODE_RECHECK_RETRY_MS,
	createModeRequirementsRecheck,
	type ModeRecheckTarget
} from './modeRequirementsRecheck';

const TARGET: ModeRecheckTarget = { tabId: 't1', presetId: 'ltx', mode: 'day_to_night' };

interface World {
	target: ModeRecheckTarget | null;
	blocked: boolean;
	becomesUnblockedAtReload: number | null;
	reloads: ModeRecheckTarget[];
}

function setup(overrides: Partial<World> = {}) {
	const world: World = {
		target: TARGET,
		blocked: true,
		becomesUnblockedAtReload: null,
		reloads: [],
		...overrides
	};
	const recheck = createModeRequirementsRecheck({
		target: () => world.target,
		isBlocked: () => world.blocked,
		reload: async (target) => {
			world.reloads.push(target);
			if (world.becomesUnblockedAtReload !== null && world.reloads.length >= world.becomesUnblockedAtReload) {
				world.blocked = false;
			}
		}
	});
	return { world, recheck };
}

beforeEach(() => {
	vi.useFakeTimers();
});

afterEach(() => {
	vi.useRealTimers();
});

describe('createModeRequirementsRecheck', () => {
	it('reloads the selected mode once and stops when that clears the block', async () => {
		const { world, recheck } = setup({ becomesUnblockedAtReload: 1 });
		await recheck.run();
		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS * 3);
		expect(world.reloads).toEqual([TARGET]);
	});

	it('never reloads when the selected mode is not blocked', async () => {
		const { world, recheck } = setup({ blocked: false });
		await recheck.run();
		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS * 3);
		expect(world.reloads).toEqual([]);
	});

	it('never reloads without a selected preset and mode', async () => {
		const { world, recheck } = setup({ target: null });
		await recheck.run();
		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS * 3);
		expect(world.reloads).toEqual([]);
	});

	it('retries exactly once when the first reload still reports the file missing', async () => {
		const { world, recheck } = setup({ becomesUnblockedAtReload: 2 });
		await recheck.run();
		expect(world.reloads).toHaveLength(1);

		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS - 1);
		expect(world.reloads).toHaveLength(1);
		await vi.advanceTimersByTimeAsync(1);
		expect(world.reloads).toHaveLength(2);

		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS * 5);
		expect(world.reloads).toHaveLength(2);
	});

	it('does not retry more than once when the file is still missing', async () => {
		const { world, recheck } = setup();
		await recheck.run();
		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS * 5);
		expect(world.reloads).toHaveLength(2);
	});

	it('aims the retry at whichever mode is selected when it fires', async () => {
		const { world, recheck } = setup();
		await recheck.run();
		world.target = { ...TARGET, mode: 'txt2vid' };
		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS * 2);
		expect(world.reloads.map((reload) => reload.mode)).toEqual(['day_to_night', 'txt2vid']);
	});

	it('does not schedule a retry when the selection moved while the reload was in flight', async () => {
		const world: World = { target: TARGET, blocked: true, becomesUnblockedAtReload: null, reloads: [] };
		const recheck = createModeRequirementsRecheck({
			target: () => world.target,
			isBlocked: () => world.blocked,
			reload: async (target) => {
				world.reloads.push(target);
				world.target = { ...TARGET, mode: 'txt2vid' };
			}
		});
		await recheck.run();
		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS * 3);
		expect(world.reloads).toHaveLength(1);
	});

	it('skips the retry when the mode stopped being blocked by itself', async () => {
		const { world, recheck } = setup();
		await recheck.run();
		world.blocked = false;
		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS * 2);
		expect(world.reloads).toHaveLength(1);
	});

	it('a newer run replaces the pending retry instead of stacking another', async () => {
		const { world, recheck } = setup();
		await recheck.run();
		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS - 100);
		await recheck.run();
		expect(world.reloads).toHaveLength(2);
		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS * 5);
		expect(world.reloads).toHaveLength(3);
	});

	it('two overlapping runs leave a single retry', async () => {
		const { world, recheck } = setup();
		await Promise.all([recheck.run(), recheck.run()]);
		expect(world.reloads).toHaveLength(2);
		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS * 5);
		expect(world.reloads).toHaveLength(3);
	});

	it('dispose cancels a pending retry and ignores later runs', async () => {
		const { world, recheck } = setup();
		await recheck.run();
		recheck.dispose();
		await vi.advanceTimersByTimeAsync(MODE_RECHECK_RETRY_MS * 3);
		await recheck.run();
		expect(world.reloads).toHaveLength(1);
	});
});
