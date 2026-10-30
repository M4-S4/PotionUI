import { describe, it, expect, vi, afterEach } from 'vitest';

const readiness = vi.hoisted(() => ({ release: null as null | (() => void) }));

vi.mock('$lib/services/websocket', async (importOriginal) => {
	const real = await importOriginal<Record<string, unknown>>();
	return {
		...real,
		createGenerationSocket: () => ({
			onConnectionChange() {},
			connect() {},
			disconnect() {},
			on() {},
			off() {},
			send() {},
			subscribe() {}
		})
	};
});

vi.mock('$lib/services/api', async (importOriginal) => {
	const real = await importOriginal<Record<string, unknown>>();
	const api = new Proxy(
		{},
		{
			get: (_target, key: string) => async () => {
				if (key === 'listPresets') return { success: true, data: [] };
				if (key === 'getReadiness') {
					await new Promise<void>((resolve) => (readiness.release = resolve));
					return {
						overall: 'not_ready',
						checks: [
							{
								area: 'content',
								status: 'not_ready',
								code: 'NO_PRESETS_ASSIGNED',
								message: 'No presets are installed and assigned to this user, so there is nothing to generate.',
								action: 'Open Admin → Presets, install a preset and assign it to this user.'
							}
						]
					};
				}
				return { success: false, error: 'unused', data: null };
			}
		}
	);
	return { ...real, api };
});

Object.defineProperty(window, 'innerWidth', { value: 1280, configurable: true });

const { mount, unmount } = await import('svelte');
const { get } = await import('svelte/store');
const { tabsStore } = await import('$lib/stores/tabs');
const { viewportWidth } = await import('$lib/stores/viewport');
const { registerBuiltinFieldComponents } = await import('$lib/fields/builtin');
const { invalidatePresets } = await import('$lib/stores/presetsCatalog');
const { default: Page } = await import('../../src/routes/generate/+page.svelte');

registerBuiltinFieldComponents();

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

let teardown: (() => void) | undefined;

afterEach(() => {
	teardown?.();
	teardown = undefined;
	readiness.release = null;
});

describe('generate page empty state', () => {
	it('renders no empty-state variant until the readiness it depends on has loaded', async () => {
		viewportWidth.set(1280);
		invalidatePresets();
		const firstId = get(tabsStore).tabs[0].id;
		tabsStore.updateTab(firstId, { selectedPreset: '', selectedMode: '' });
		const target = document.body.appendChild(document.createElement('div'));
		const app = mount(Page as never, { target, props: {} });
		teardown = () => {
			unmount(app);
			target.remove();
		};

		for (let i = 0; i < 100 && !readiness.release; i++) await wait(10);
		expect(readiness.release).not.toBeNull();
		await wait(50);

		expect(target.textContent).not.toContain('Nothing to generate with yet');
		expect(target.textContent).not.toContain('nothing to generate');

		readiness.release!();
		for (let i = 0; i < 100 && !target.textContent?.includes('Nothing to generate with yet'); i++) await wait(10);

		expect(target.textContent).toContain('Nothing to generate with yet');
		expect(target.textContent).toContain('nothing to generate');
	});
});
