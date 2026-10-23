import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const gotoMock = vi.fn(async () => {});
vi.mock('$app/navigation', () => ({ goto: (...args: unknown[]) => gotoMock(...(args as [])) }));

const { createUrlState, hrefWithQuery } = await import('./urlState');

type Overrides = { q?: string };

function make(delay?: number) {
	return createUrlState<Overrides>({ href: (o) => `/admin?q=${o.q ?? ''}`, delay });
}

beforeEach(() => {
	vi.useFakeTimers();
	gotoMock.mockClear();
});

afterEach(() => {
	vi.useRealTimers();
});

describe('hrefWithQuery', () => {
	it('omits the question mark for empty params', () => {
		expect(hrefWithQuery('/a', new URLSearchParams())).toBe('/a');
		expect(hrefWithQuery('/a', new URLSearchParams({ x: '1' }))).toBe('/a?x=1');
	});
});

describe('createUrlState', () => {
	it('go navigates with no options by default', async () => {
		await make().go({ q: 'a' });
		expect(gotoMock).toHaveBeenCalledWith('/admin?q=a', {});
	});

	it('go forwards only the requested options', async () => {
		await make().go({ q: 'a' }, { keepFocus: true, noScroll: true });
		expect(gotoMock).toHaveBeenCalledWith('/admin?q=a', { keepFocus: true, noScroll: true });
	});

	it('replace uses replaceState, keepFocus and noScroll', async () => {
		await make().replace({ q: 'b' });
		expect(gotoMock).toHaveBeenCalledWith('/admin?q=b', { replaceState: true, keepFocus: true, noScroll: true });
	});

	it('debounced waits the delay and navigates once with the latest overrides', () => {
		const state = make();
		state.debounced({ q: 'a' });
		state.debounced({ q: 'ab' });
		vi.advanceTimersByTime(249);
		expect(gotoMock).not.toHaveBeenCalled();
		vi.advanceTimersByTime(1);
		expect(gotoMock).toHaveBeenCalledTimes(1);
		expect(gotoMock).toHaveBeenCalledWith('/admin?q=ab', { replaceState: true, keepFocus: true, noScroll: true });
	});

	it('debounced honours a custom delay', () => {
		const state = make(200);
		state.debounced({ q: 'a' });
		vi.advanceTimersByTime(200);
		expect(gotoMock).toHaveBeenCalledTimes(1);
	});

	it('debounced runs onFire before navigating', () => {
		const order: string[] = [];
		gotoMock.mockImplementationOnce(async () => {
			order.push('goto');
		});
		const state = make();
		state.debounced({ q: 'a' }, () => order.push('fire'));
		vi.advanceTimersByTime(250);
		expect(order).toEqual(['fire', 'goto']);
	});

	it('cancel drops a pending navigation', () => {
		const onFire = vi.fn();
		const state = make();
		state.debounced({ q: 'a' }, onFire);
		state.cancel();
		vi.advanceTimersByTime(1000);
		expect(gotoMock).not.toHaveBeenCalled();
		expect(onFire).not.toHaveBeenCalled();
	});

	it('resolves href at fire time', () => {
		let tab = 'x';
		const state = createUrlState<Overrides>({ href: () => `/t/${tab}` });
		state.debounced({});
		tab = 'y';
		vi.advanceTimersByTime(250);
		expect(gotoMock).toHaveBeenCalledWith('/t/y', expect.any(Object));
	});
});
