// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import MountProbe from './fixtures/MountProbe.svelte';
import { mountComponent } from './helpers/mount';
import { goto } from './stubs/appNavigation';
import { page } from './stubs/appStores';
import { get } from 'svelte/store';

describe('mountComponent', () => {
	it('renders props and updates them on rerender', () => {
		const view = mountComponent(MountProbe, { label: 'one' });
		expect(view.target.querySelector('[data-testid="probe"]')!.textContent).toBe('one');
		view.rerender({ label: 'two' });
		expect(view.target.querySelector('[data-testid="probe"]')!.textContent).toBe('two');
	});

	it('removes the previous test mount without an explicit destroy', () => {
		expect(document.querySelector('[data-testid="probe"]')).toBeNull();
	});

	it('destroy is idempotent and removes the target', () => {
		const view = mountComponent(MountProbe);
		view.destroy();
		view.destroy();
		expect(document.body.contains(view.target)).toBe(false);
	});
});

describe('shared navigation stub', () => {
	it('goto updates the page url', async () => {
		await goto('/history?sort=largest');
		const url = get(page).url;
		expect(url.pathname).toBe('/history');
		expect(url.search).toBe('?sort=largest');
	});

	it('starts every test from the default url', () => {
		expect(get(page).url.pathname).toBe('/generate');
		expect(get(page).url.search).toBe('');
	});
});
