// @vitest-environment jsdom
//
// Drives the REAL compiled comfyui-backend plugin dist for the "Imported
// presets" row actions: reload (spinner while in flight, then an inline
// lint-result chip), edit (writes the cross-tab sessionStorage handoff and
// dispatches the generic potionui:switch-plugin-tab event the host listens
// for), and delete (hand-built confirm modal, Esc/Enter, row disappears on
// confirm). Same stage/load approach as importedPresetsTab.test.ts.
import { describe, expect, it, vi, beforeAll, afterAll, afterEach } from 'vitest';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { stagePluginDist } from './helpers/stagePluginDist';
import { installPluginHost, removePluginHost, type PluginHostStubs } from './helpers/installPluginHost';
import { mount, unmount } from 'svelte';
import { _wrapPluginDistComponent } from '$lib/plugin-api/componentResolver';

const REPO_ROOT = resolve(__dirname, '../../..');
const DIST_PATH = 'content/plugins/marketplace/comfyui-backend/frontend/dist/ImportedPresetsTab.js';

async function loadDist(): Promise<any> {
	const mod = await import(/* @vite-ignore */ stagePluginDist(resolve(REPO_ROOT, DIST_PATH)));
	return mod.default;
}

function target(): HTMLDivElement {
	const el = document.createElement('div');
	document.body.appendChild(el);
	return el;
}

async function settle() {
	await new Promise((r) => setTimeout(r, 0));
	await new Promise((r) => setTimeout(r, 0));
}

function jsonResponse(body: unknown, ok = true, status = ok ? 200 : 400) {
	return { ok, status, json: async () => body } as Response;
}

const PRESET_ROW = {
	preset_id: 'P1',
	name: 'LTX 2.5 — My workflow',
	family: 'LTX',
	variant: 'imported',
	format: 'api',
	has_sidecar: true,
	requirements_summary: { ok: 6, missing: 0, unknown: 0, optional_missing: 0 },
	created_at: Math.floor(Date.now() / 1000) - 120
};

let ImportedPresetsTab: any;
let host: PluginHostStubs;

beforeAll(async () => {
	host = await installPluginHost();
	if (!existsSync(resolve(REPO_ROOT, DIST_PATH))) {
		throw new Error(`${DIST_PATH} is missing - run \`node scripts/build-plugins.mjs comfyui-backend\` first.`);
	}
	const raw = await loadDist();
	ImportedPresetsTab = _wrapPluginDistComponent(raw);
});

afterAll(removePluginHost);

afterEach(() => {
	vi.unstubAllGlobals();
	document.body.innerHTML = '';
	sessionStorage.clear();
});

describe('ImportedPresetsTab row actions (real compiled dist)', () => {
	it('reload shows a spinner then an inline lint-result chip, and refreshes the list', async () => {
		let reloadCalled = false;
		const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
			if (url === '/api/plugins/comfyui-backend/presets/imported') {
				return jsonResponse({ presets: [reloadCalled ? { ...PRESET_ROW, name: 'Reloaded name' } : PRESET_ROW] });
			}
			if (url === '/api/plugins/comfyui-backend/presets/imported/P1/reload' && init?.method === 'POST') {
				reloadCalled = true;
				return jsonResponse({ preset_id: 'P1', path: 'content/presets/local/LTX/imported', mode: 'txt2img', lint: { errors: [], warnings: [] } });
			}
			throw new Error(`Unexpected fetch: ${url}`);
		});
		vi.stubGlobal('fetch', fetchMock);

		const el = target();
		const instance = mount(ImportedPresetsTab, { target: el, props: { pluginId: 'comfyui-backend' } });
		await settle();

		const reloadBtn = el.querySelector<HTMLButtonElement>('button[aria-label="Reload from source"]')!;
		expect(reloadBtn).toBeTruthy();
		reloadBtn.click();
		await settle();

		const resultLine = el.querySelector('.ip-reload-result');
		expect(resultLine?.textContent).toContain('lint clean');
		// The row's own data refreshed after reload.
		expect(el.textContent).toContain('Reloaded name');

		unmount(instance);
	});

	it('reload shows the danger tone on a lint error', async () => {
		const fetchMock = vi.fn(async (url: string) => {
			if (url === '/api/plugins/comfyui-backend/presets/imported') return jsonResponse({ presets: [PRESET_ROW] });
			if (url === '/api/plugins/comfyui-backend/presets/imported/P1/reload') {
				return jsonResponse({ preset_id: 'P1', path: 'x', mode: 'txt2img', lint: { errors: ['bad requirement'], warnings: [] } });
			}
			throw new Error(`Unexpected fetch: ${url}`);
		});
		vi.stubGlobal('fetch', fetchMock);

		const el = target();
		const instance = mount(ImportedPresetsTab, { target: el, props: { pluginId: 'comfyui-backend' } });
		await settle();

		el.querySelector<HTMLButtonElement>('button[aria-label="Reload from source"]')!.click();
		await settle();

		const resultLine = el.querySelector('.ip-reload-result');
		expect(resultLine?.querySelector('[role="alert"]')).toBeTruthy();
		expect(resultLine?.textContent).toContain('1 lint error');

		unmount(instance);
	});

	it('edit stashes the preset id in sessionStorage and asks the host to switch tabs', async () => {
		const fetchMock = vi.fn(async (url: string) => {
			if (url === '/api/plugins/comfyui-backend/presets/imported') return jsonResponse({ presets: [PRESET_ROW] });
			throw new Error(`Unexpected fetch: ${url}`);
		});
		vi.stubGlobal('fetch', fetchMock);

		const switchHandler = vi.fn();
		window.addEventListener('potionui:switch-plugin-tab', switchHandler);

		const el = target();
		const instance = mount(ImportedPresetsTab, { target: el, props: { pluginId: 'comfyui-backend' } });
		await settle();

		el.querySelector<HTMLButtonElement>('button[aria-label="Edit"]')!.click();
		await settle();

		expect(sessionStorage.getItem('comfyui-import-edit-preset-id')).toBe('P1');
		expect(switchHandler).toHaveBeenCalledTimes(1);
		const event = switchHandler.mock.calls[0][0] as CustomEvent;
		expect(event.detail).toEqual({ pluginId: 'comfyui-backend', tabId: 'import-workflow' });

		window.removeEventListener('potionui:switch-plugin-tab', switchHandler);
		unmount(instance);
	});

	it('delete asks the host to confirm; declining sends nothing, accepting removes the row', async () => {
		let deleted = false;
		const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
			if (url === '/api/plugins/comfyui-backend/presets/imported') {
				return jsonResponse({ presets: deleted ? [] : [PRESET_ROW] });
			}
			if (url === '/api/plugins/comfyui-backend/presets/imported/P1' && init?.method === 'DELETE') {
				deleted = true;
				return jsonResponse({ deleted: true, preset_id: 'P1' });
			}
			throw new Error(`Unexpected fetch: ${url}`);
		});
		vi.stubGlobal('fetch', fetchMock);

		const el = target();
		const instance = mount(ImportedPresetsTab, { target: el, props: { pluginId: 'comfyui-backend' } });
		await settle();

		host.confirm.mockResolvedValueOnce(false);
		el.querySelector<HTMLButtonElement>('button[aria-label="Delete"]')!.click();
		await settle();

		expect(host.confirm).toHaveBeenCalledTimes(1);
		expect(host.confirm.mock.calls[0][0]).toMatchObject({ variant: 'danger' });
		expect(host.confirm.mock.calls[0][0].title).toContain('LTX 2.5 — My workflow');
		expect(fetchMock.mock.calls.some((c) => c[1]?.method === 'DELETE')).toBe(false);
		expect(el.querySelector('[data-imported-presets]')).toBeTruthy();

		el.querySelector<HTMLButtonElement>('button[aria-label="Delete"]')!.click();
		await settle();

		expect(fetchMock.mock.calls.some((c) => c[1]?.method === 'DELETE')).toBe(true);
		expect(el.querySelector('[data-imported-presets]')).toBeNull();
		expect(el.textContent).toContain('No presets have been imported yet');
		expect(host.toast).not.toHaveBeenCalled();

		unmount(instance);
	});

	it('a delete in flight disables the row button and a second click sends no second request', async () => {
		host.confirm.mockClear();
		let release: (value: Response) => void = () => {};
		const pending = new Promise<Response>((resolveResponse) => {
			release = resolveResponse;
		});
		const fetchMock = vi.fn((url: string, init?: RequestInit) => {
			if (url === '/api/plugins/comfyui-backend/presets/imported') return Promise.resolve(jsonResponse({ presets: [PRESET_ROW] }));
			if (init?.method === 'DELETE') return pending;
			throw new Error(`Unexpected fetch: ${url}`);
		});
		vi.stubGlobal('fetch', fetchMock);

		const el = target();
		const instance = mount(ImportedPresetsTab, { target: el, props: { pluginId: 'comfyui-backend' } });
		await settle();

		const deleteButton = el.querySelector<HTMLButtonElement>('button[aria-label="Delete"]')!;
		deleteButton.click();
		await settle();

		const deleteCalls = () => fetchMock.mock.calls.filter((c) => c[1]?.method === 'DELETE').length;
		expect(deleteCalls()).toBe(1);
		expect(el.querySelector<HTMLButtonElement>('button[aria-label="Delete"]')!.disabled).toBe(true);

		el.querySelector<HTMLButtonElement>('button[aria-label="Delete"]')!.click();
		await settle();
		expect(deleteCalls()).toBe(1);
		expect(host.confirm).toHaveBeenCalledTimes(1);

		release(jsonResponse({ deleted: true, preset_id: 'P1' }));
		await settle();
		expect(el.querySelector('[data-imported-presets]')).toBeNull();

		unmount(instance);
	});

	it('a failed delete keeps the row and shows the failure as a toast', async () => {
		const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
			if (url === '/api/plugins/comfyui-backend/presets/imported') return jsonResponse({ presets: [PRESET_ROW] });
			if (init?.method === 'DELETE') return jsonResponse({ detail: 'Preset is in use' }, false, 409);
			throw new Error(`Unexpected fetch: ${url}`);
		});
		vi.stubGlobal('fetch', fetchMock);

		const el = target();
		const instance = mount(ImportedPresetsTab, { target: el, props: { pluginId: 'comfyui-backend' } });
		await settle();

		el.querySelector<HTMLButtonElement>('button[aria-label="Delete"]')!.click();
		await settle();

		expect(host.toast).toHaveBeenCalledWith('error', 'Preset is in use');
		expect(el.querySelector('[data-imported-presets]')).toBeTruthy();

		unmount(instance);
	});
});
