import { afterEach, describe, expect, it, vi } from 'vitest';
import { mountComponent } from './helpers/mount';
import { resetScrollOnChange } from '../../src/lib/utils/scrollToTop';

const { default: SetupFocusShell } = await import('../../src/lib/components/setup/SetupFocusShell.svelte');

afterEach(() => {
	vi.restoreAllMocks();
	document.body.innerHTML = '';
});

function scroller() {
	const el = document.createElement('div');
	el.style.overflowY = 'auto';
	let top = 350;
	Object.defineProperty(el, 'scrollTop', { get: () => top, set: (v: number) => (top = v), configurable: true });
	document.body.appendChild(el);
	return el;
}

describe('resetScrollOnChange', () => {
	it('scrolls to the top only when the watched value changes', () => {
		const scrollTo = vi.spyOn(window, 'scrollTo').mockImplementation(() => {});
		const host = scroller();
		const node = host.appendChild(document.createElement('div'));
		const action = resetScrollOnChange(node, 'starter');

		action.update('starter');
		expect(host.scrollTop).toBe(350);
		expect(scrollTo).not.toHaveBeenCalled();

		action.update('install');
		expect(host.scrollTop).toBe(0);
		expect(scrollTo).toHaveBeenCalledOnce();
	});
});

describe('SetupFocusShell step changes', () => {
	it('scrolls the container and the window to the top when the step changes', () => {
		const scrollTo = vi.spyOn(window, 'scrollTo').mockImplementation(() => {});
		const host = scroller();
		const view = mountComponent(SetupFocusShell, { current: 'welcome' }, { target: host });
		expect(host.scrollTop).toBe(350);

		view.rerender({ current: 'account' });

		expect(host.scrollTop).toBe(0);
		expect(scrollTo).toHaveBeenCalledWith({ top: 0, left: 0, behavior: 'instant' });
	});

	it('leaves the scroll offset alone when the step is unchanged', () => {
		const scrollTo = vi.spyOn(window, 'scrollTo').mockImplementation(() => {});
		const host = scroller();
		const view = mountComponent(SetupFocusShell, { current: 'welcome' }, { target: host });

		view.rerender({ current: 'welcome' });

		expect(host.scrollTop).toBe(350);
		expect(scrollTo).not.toHaveBeenCalled();
	});
});
