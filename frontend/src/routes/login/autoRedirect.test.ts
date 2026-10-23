import { describe, it, expect } from 'vitest';
import { AUTO_REDIRECT_GUARD_MS, pickAutoRedirectProvider, shouldAutoRedirect } from './autoRedirect';

const on = { id: 'a', label: 'A', start_path: '/api/plugins/a/start', auto_redirect: true };
const off = { id: 'b', label: 'B', start_path: '/api/plugins/b/start', auto_redirect: false };

function decide(search: string, over: Partial<Parameters<typeof shouldAutoRedirect>[0]> = {}) {
	return shouldAutoRedirect({
		providers: [on],
		params: new URLSearchParams(search),
		lastAttemptAt: null,
		now: 1_000_000,
		...over
	});
}

describe('pickAutoRedirectProvider', () => {
	it('picks the first flagged provider and ignores unflagged ones', () => {
		expect(pickAutoRedirectProvider([off, on])).toBe(on);
		expect(pickAutoRedirectProvider([off])).toBeNull();
		expect(pickAutoRedirectProvider([])).toBeNull();
	});

	it('requires a literal true', () => {
		expect(pickAutoRedirectProvider([{ ...on, auto_redirect: 'yes' as never }])).toBeNull();
	});
});

describe('shouldAutoRedirect', () => {
	it('redirects a plain visit', () => {
		expect(decide('')).toBe(on);
	});

	it('redirects when only a return path is present', () => {
		expect(decide('next=%2Fhistory')).toBe(on);
	});

	it.each(['local=1', 'error=external_login', 'expired=1', 'signedout=1', 'add=1', 'dup=x', 'as=x'])(
		'is suppressed by %s',
		(search) => {
			expect(decide(search)).toBeNull();
		}
	);

	it('is suppressed inside the guard window and allowed after it', () => {
		expect(decide('', { lastAttemptAt: 1_000_000 - 1000 })).toBeNull();
		expect(decide('', { lastAttemptAt: 1_000_000 - AUTO_REDIRECT_GUARD_MS })).toBe(on);
	});

	it('ignores a guard timestamp from the future', () => {
		expect(decide('', { lastAttemptAt: 1_000_000 + 5000 })).toBe(on);
	});

	it('does nothing without a flagged provider', () => {
		expect(decide('', { providers: [off] })).toBeNull();
	});
});
