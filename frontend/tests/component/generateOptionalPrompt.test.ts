import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';

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

const MODES = ['recast', 'restyle', 'txt2img', 'upscale'];

vi.mock('$lib/services/api', async (importOriginal) => {
	const real = await importOriginal<Record<string, unknown>>();
	const api = new Proxy(
		{},
		{
			get: (_target, key: string) => async () => {
				if (key === 'getPresetFormSchema') return { success: true, data: { form_schema: { properties: {} } } };
				if (key === 'getPresetModes') {
					return {
						success: true,
						data: {
							modes: MODES.map((name) => ({ name, label: name, variants: [], requirements_missing: [] })),
							default_mode: 'txt2img'
						}
					};
				}
				if (key === 'getPreset') {
					return {
						success: true,
						data: {
							vars: {
								optional_prompt_modes: [
									'recast',
									{ mode: 'restyle', when: { field: 'restyle_task', in: ['colourise', 'restore'] } },
									'upscale'
								],
								promptless_modes: ['upscale']
							},
							styles: []
						}
					};
				}
				if (key === 'listPresets') {
					return {
						success: true,
						data: [{ id: 'recaster', name: 'Recaster', version: '1', engine: 'native', modes: MODES, engine_available: true }]
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

beforeEach(() => {
	tabsStore.reset();
});

afterEach(() => {
	teardown?.();
	teardown = undefined;
});

async function render(mode: string, formData: Record<string, unknown> = {}) {
	invalidatePresets();
	tabsStore.updateTab(get(tabsStore).tabs[0].id, { selectedPreset: 'recaster', selectedMode: mode, formData, prompt: '' });
	const target = document.body.appendChild(document.createElement('div'));
	const app = mount(Page as never, { target, props: {} });
	teardown = () => {
		unmount(app);
		target.remove();
	};
	for (let i = 0; i < 100 && !target.textContent?.includes('Recaster'); i++) await wait(10);
	await wait(80);
	return target;
}

function generateButton(target: HTMLElement) {
	return target.querySelector('button.generate-button') as HTMLButtonElement;
}

function promptPane(target: HTMLElement) {
	return target.querySelector('.prompt-composer');
}

describe('Generate in a mode whose prompt is optional', () => {
	it('keeps the prompt pane and enables Generate with no prompt text', async () => {
		const target = await render('recast');

		expect(promptPane(target)).not.toBeNull();
		expect(generateButton(target).disabled).toBe(false);
		expect(target.textContent).not.toContain('Missing prompt');
	});

	it('still asks for a prompt in a mode that is not listed', async () => {
		const target = await render('txt2img');

		expect(promptPane(target)).not.toBeNull();
		expect(generateButton(target).disabled).toBe(true);
		expect(target.textContent).toContain('Missing prompt');
	});

	it('follows the form value of a conditional entry', async () => {
		const colourise = await render('restyle', { restyle_task: 'colourise' });
		expect(promptPane(colourise)).not.toBeNull();
		expect(generateButton(colourise).disabled).toBe(false);
		teardown?.();

		const custom = await render('restyle', { restyle_task: 'custom' });
		expect(promptPane(custom)).not.toBeNull();
		expect(generateButton(custom).disabled).toBe(true);
		expect(custom.textContent).toContain('Missing prompt');
	});

	it('hides the pane when the mode is promptless as well', async () => {
		const target = await render('upscale');

		expect(promptPane(target)).toBeNull();
		expect(generateButton(target).disabled).toBe(false);
	});
});
