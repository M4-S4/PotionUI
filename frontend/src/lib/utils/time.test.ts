import { describe, it, expect, vi, afterEach } from 'vitest';
import { debounce, sleep, startOfDay } from './time';

afterEach(() => {
	vi.useRealTimers();
});

describe('startOfDay', () => {
	it('drops the time of day in local time', () => {
		const date = new Date(2026, 6, 3, 18, 45, 12, 500);
		expect(startOfDay(date)).toBe(new Date(2026, 6, 3).getTime());
	});
});

describe('sleep', () => {
	it('resolves only after the delay', async () => {
		vi.useFakeTimers();
		const done = vi.fn();
		void sleep(100).then(done);
		await vi.advanceTimersByTimeAsync(99);
		expect(done).not.toHaveBeenCalled();
		await vi.advanceTimersByTimeAsync(1);
		expect(done).toHaveBeenCalledOnce();
	});
});

describe('debounce', () => {
	it('runs once with the last arguments after a burst', () => {
		vi.useFakeTimers();
		const fn = vi.fn();
		const debounced = debounce(fn, 50);
		debounced('a');
		vi.advanceTimersByTime(30);
		debounced('b');
		vi.advanceTimersByTime(49);
		expect(fn).not.toHaveBeenCalled();
		vi.advanceTimersByTime(1);
		expect(fn).toHaveBeenCalledOnce();
		expect(fn).toHaveBeenCalledWith('b');
	});

	it('cancel drops the pending call', () => {
		vi.useFakeTimers();
		const fn = vi.fn();
		const debounced = debounce(fn, 50);
		debounced();
		debounced.cancel();
		vi.advanceTimersByTime(200);
		expect(fn).not.toHaveBeenCalled();
		debounced();
		vi.advanceTimersByTime(50);
		expect(fn).toHaveBeenCalledOnce();
	});
});
