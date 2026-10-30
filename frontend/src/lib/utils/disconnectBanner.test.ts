import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createDisconnectBanner } from './disconnectBanner';

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

function setup() {
	const shown: boolean[] = [];
	const banner = createDisconnectBanner((visible) => shown.push(visible), 1000);
	return { shown, banner };
}

describe('createDisconnectBanner', () => {
	it('stays hidden while the first connection is still being made', () => {
		const { shown, banner } = setup();
		banner.connecting();

		vi.advanceTimersByTime(900);

		expect(shown).toEqual([]);
	});

	it('never shows when the socket connects within the grace period', () => {
		const { shown, banner } = setup();
		banner.connecting();
		vi.advanceTimersByTime(500);

		banner.changed(true);
		vi.advanceTimersByTime(5000);

		expect(shown).toEqual([false]);
	});

	it('shows after the grace period when the connection never comes up', () => {
		const { shown, banner } = setup();
		banner.connecting();

		vi.advanceTimersByTime(1000);

		expect(shown).toEqual([true]);
	});

	it('shows a real disconnect only after the grace period and hides again on reconnect', () => {
		const { shown, banner } = setup();
		banner.changed(true);
		banner.changed(false);
		vi.advanceTimersByTime(999);
		expect(shown).toEqual([false]);

		vi.advanceTimersByTime(1);
		expect(shown).toEqual([false, true]);

		banner.changed(true);
		expect(shown).toEqual([false, true, false]);
	});

	it('shows nothing after it is destroyed', () => {
		const { shown, banner } = setup();
		banner.connecting();
		banner.destroy();

		vi.advanceTimersByTime(5000);

		expect(shown).toEqual([]);
	});
});
