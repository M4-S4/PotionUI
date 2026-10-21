import { vi } from 'vitest';
import { flushSync } from 'svelte';

export async function settle(rounds = 8): Promise<void> {
	for (let i = 0; i < rounds; i++) {
		if (vi.isFakeTimers()) await vi.advanceTimersByTimeAsync(0);
		else await new Promise((resolve) => setTimeout(resolve, 0));
	}
}

export async function settleFlush(rounds = 5): Promise<void> {
	await settle(rounds);
	flushSync();
}

export async function advanceTimers(ms: number): Promise<void> {
	await vi.advanceTimersByTimeAsync(ms);
	flushSync();
}
