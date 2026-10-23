import { describe, it, expect, vi, afterEach } from 'vitest';

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

const catalog = vi.hoisted(() => ({ engineAvailable: false }));

vi.mock('$lib/services/api', async (importOriginal) => {
	const real = await importOriginal<Record<string, unknown>>();
	const api = new Proxy(
		{},
		{
			get: (_target, key: string) => async () => {
				if (key === 'getPresetFormSchema') return { success: true, data: { form_schema: { properties: {} } } };
				if (key === 'getPresetModes') {
					return { success: true, data: { modes: [{ name: 'image', label: 'Image', variants: [] }], default_mode: 'image' } };
				}
				if (key === 'getPreset') return { success: true, data: { vars: { promptless_modes: ['image'] }, styles: [] } };
				if (key === 'listPresets') {
					return {
						success: true,
						data: [{ id: 'flux1', name: 'FLUX.1 Dev', version: '1', engine: 'comfyui', modes: ['image'], engine_available: catalog.engineAvailable }]
					};
				}
				if (key === 'getReadiness') return { checks: [], ready: true, status: 'ready' };
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
const { registerBuiltinFieldComponents } = await import('$lib/fields/builtin');
const { invalidatePresets } = await import('$lib/stores/presetsCatalog');
const { default: Page } = await import('../../src/routes/generate/+page.svelte');

registerBuiltinFieldComponents();

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

let teardown: (() => void) | undefined;

afterEach(() => {
	teardown?.();
	teardown = undefined;
});

async function render() {
	invalidatePresets();
	const firstId = get(tabsStore).tabs[0].id;
	tabsStore.updateTab(firstId, { selectedPreset: 'flux1', selectedMode: 'image' });
	const target = document.body.appendChild(document.createElement('div'));
	const app = mount(Page as never, { target, props: {} });
	teardown = () => {
		unmount(app);
		target.remove();
	};
	for (let i = 0; i < 100 && !target.textContent?.includes('FLUX.1 Dev'); i++) await wait(10);
	await wait(50);
	return target;
}

describe('Generate with a preset that cannot run', () => {
	it('stays disabled and says why when no backend serves the preset engine', async () => {
		catalog.engineAvailable = false;
		const target = await render();
		expect(target.textContent).toContain("Can't generate yet");
		expect(target.textContent).toContain('No ComfyUI backend is set up');
		expect((target.querySelector('button.generate-button') as HTMLButtonElement).disabled).toBe(true);
	});

	it('enables Generate once a backend serves the preset engine', async () => {
		catalog.engineAvailable = true;
		const target = await render();
		expect(target.textContent).toContain('FLUX.1 Dev');
		expect(target.textContent).not.toContain('No ComfyUI backend is set up');
		expect((target.querySelector('button.generate-button') as HTMLButtonElement).disabled).toBe(false);
	});
});
