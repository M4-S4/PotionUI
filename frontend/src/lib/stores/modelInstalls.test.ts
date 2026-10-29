import { describe, it, expect, vi } from 'vitest';

vi.mock('$lib/utils/logger', () => ({
	logger: { error: vi.fn(), warn: vi.fn(), info: vi.fn(), debug: vi.fn() }
}));

const { announceModelInstalled, onModelInstalled } = await import('./modelInstalls');
const { logger } = await import('$lib/utils/logger');

describe('modelInstalls', () => {
	it('tells every listener, and stops telling one that unsubscribed', () => {
		const first = vi.fn();
		const second = vi.fn();
		const offFirst = onModelInstalled(first);
		const offSecond = onModelInstalled(second);

		announceModelInstalled();
		expect(first).toHaveBeenCalledTimes(1);
		expect(second).toHaveBeenCalledTimes(1);

		offFirst();
		announceModelInstalled();
		expect(first).toHaveBeenCalledTimes(1);
		expect(second).toHaveBeenCalledTimes(2);
		offSecond();
	});

	it('keeps notifying the rest when one listener throws', () => {
		const broken = vi.fn(() => {
			throw new Error('boom');
		});
		const healthy = vi.fn();
		const offBroken = onModelInstalled(broken);
		const offHealthy = onModelInstalled(healthy);

		expect(() => announceModelInstalled()).not.toThrow();
		expect(healthy).toHaveBeenCalledTimes(1);
		expect(logger.error).toHaveBeenCalled();
		offBroken();
		offHealthy();
	});

	it('is a no-op with no listeners', () => {
		expect(() => announceModelInstalled()).not.toThrow();
	});
});
