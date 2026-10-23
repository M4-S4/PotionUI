import { describe, it, expect, afterEach } from 'vitest';
import { get } from 'svelte/store';
import { tabSessionDirty, setTabSessionDirty, clearTabSessionDirty } from './tabSessionDirty';

afterEach(() => {
	clearTabSessionDirty('a');
	clearTabSessionDirty('b');
});

describe('tabSessionDirty', () => {
	it('records each tab on its own', () => {
		setTabSessionDirty('a', true);
		setTabSessionDirty('b', false);
		expect(get(tabSessionDirty)).toEqual({ a: true, b: false });
	});

	it('notifies only when a tab flips, not when the same value is set again', () => {
		const seen: Record<string, boolean>[] = [];
		const stop = tabSessionDirty.subscribe((value) => seen.push(value));
		setTabSessionDirty('a', true);
		setTabSessionDirty('a', true);
		setTabSessionDirty('a', true);
		setTabSessionDirty('a', false);
		stop();
		expect(seen.map((value) => value.a)).toEqual([undefined, true, false]);
	});

	it('forgets a tab on clear and ignores tabs it never saw', () => {
		setTabSessionDirty('a', true);
		const seen: Record<string, boolean>[] = [];
		const stop = tabSessionDirty.subscribe((value) => seen.push(value));
		clearTabSessionDirty('a');
		clearTabSessionDirty('never');
		stop();
		expect(seen).toHaveLength(2);
		expect('a' in get(tabSessionDirty)).toBe(false);
	});
});
