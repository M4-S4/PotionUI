import { beforeEach, describe, expect, it, vi } from 'vitest';
import { writable } from 'svelte/store';
import { CORE_TOOLS } from '../../../tests/component/fixtures/mediaToolCatalog';

const listMediaTools = vi.hoisted(() => vi.fn());
const toastShow = vi.hoisted(() => vi.fn());

vi.mock('$lib/services/api/mediaTools', () => ({
	listMediaTools: (...args: unknown[]) => listMediaTools(...args)
}));
vi.mock('$lib/stores/toast', () => ({ toasts: { show: (...args: unknown[]) => toastShow(...args) } }));
vi.mock('$lib/plugin-api/extensionRefresh', () => ({ extensionRegistrations: writable(0) }));

async function fresh() {
	vi.resetModules();
	const pluginTools = await import('./pluginTools');
	const tools = await import('./tools');
	return { ...pluginTools, listTools: tools.listTools };
}

beforeEach(() => {
	listMediaTools.mockReset();
	toastShow.mockReset();
});

describe('loading the catalog for the menus', () => {
	it('puts the catalog tools on the menus and says nothing when it loads', async () => {
		listMediaTools.mockResolvedValue(CORE_TOOLS);
		const { loadPluginMediaTools, listTools } = await fresh();

		await loadPluginMediaTools();

		expect(listTools().map((tool) => tool.id)).toContain('trim');
		expect(toastShow).not.toHaveBeenCalled();
	});

	it('tells the user once, without blocking, when the catalog cannot be loaded', async () => {
		listMediaTools.mockRejectedValue(new Error('offline'));
		const { loadPluginMediaTools, CATALOG_LOAD_FAILED_MESSAGE } = await fresh();

		await loadPluginMediaTools();

		expect(CATALOG_LOAD_FAILED_MESSAGE).toBe("Editing tools couldn't load");
		expect(toastShow).toHaveBeenCalledTimes(1);
		expect(toastShow).toHaveBeenCalledWith('warning', "Editing tools couldn't load");
	});

	it('does not repeat the toast when another page load fails the same way', async () => {
		listMediaTools.mockRejectedValue(new Error('offline'));
		const { loadPluginMediaTools } = await fresh();

		await loadPluginMediaTools();
		await loadPluginMediaTools();

		expect(toastShow).toHaveBeenCalledTimes(1);
	});

	it('does not toast for a failure that two callers share', async () => {
		listMediaTools.mockRejectedValue(new Error('offline'));
		const { loadPluginMediaTools } = await fresh();

		await Promise.all([loadPluginMediaTools(), loadPluginMediaTools()]);

		expect(toastShow).toHaveBeenCalledTimes(1);
	});

	it('leaves the menus without catalog tools while it is down', async () => {
		listMediaTools.mockRejectedValue(new Error('offline'));
		const { loadPluginMediaTools, listTools } = await fresh();

		await loadPluginMediaTools();

		expect(listTools().filter((tool) => !!tool.catalogToolId)).toEqual([]);
	});
});

describe('retrying when a menu opens', () => {
	it('tries again once and brings the tools in when the catalog is back', async () => {
		listMediaTools.mockRejectedValueOnce(new Error('offline')).mockResolvedValue(CORE_TOOLS);
		const { loadPluginMediaTools, retryMediaToolCatalog, listTools } = await fresh();
		await loadPluginMediaTools();
		expect(listTools().map((tool) => tool.id)).not.toContain('trim');

		await retryMediaToolCatalog();

		expect(listMediaTools).toHaveBeenCalledTimes(2);
		expect(listTools().map((tool) => tool.id)).toContain('trim');
		expect(toastShow).toHaveBeenCalledTimes(1);
	});

	it('bumps the registration counter so an open bar rebuilds its groups', async () => {
		listMediaTools.mockRejectedValueOnce(new Error('offline')).mockResolvedValue(CORE_TOOLS);
		const { loadPluginMediaTools, retryMediaToolCatalog, mediaToolRegistrations } = await fresh();
		await loadPluginMediaTools();
		const seen: number[] = [];
		mediaToolRegistrations.subscribe((tick) => seen.push(tick));

		await retryMediaToolCatalog();

		expect(seen.at(-1)).toBeGreaterThan(seen[0]);
	});

	it('retries only once while the catalog stays down, and stays quiet', async () => {
		listMediaTools.mockRejectedValue(new Error('offline'));
		const { loadPluginMediaTools, retryMediaToolCatalog } = await fresh();
		await loadPluginMediaTools();

		await retryMediaToolCatalog();
		await retryMediaToolCatalog();
		await retryMediaToolCatalog();

		expect(listMediaTools).toHaveBeenCalledTimes(2);
		expect(toastShow).toHaveBeenCalledTimes(1);
	});

	it('does nothing when the catalog loaded fine', async () => {
		listMediaTools.mockResolvedValue(CORE_TOOLS);
		const { loadPluginMediaTools, retryMediaToolCatalog } = await fresh();
		await loadPluginMediaTools();

		await retryMediaToolCatalog();

		expect(listMediaTools).toHaveBeenCalledTimes(1);
	});

	it('does nothing before any load was attempted', async () => {
		const { retryMediaToolCatalog } = await fresh();

		await retryMediaToolCatalog();

		expect(listMediaTools).not.toHaveBeenCalled();
	});

	it('does not fetch again when a second page mount follows a failed load, and still allows its one retry', async () => {
		listMediaTools.mockRejectedValue(new Error('offline'));
		const { loadPluginMediaTools, retryMediaToolCatalog } = await fresh();
		await loadPluginMediaTools();
		await loadPluginMediaTools();
		expect(listMediaTools).toHaveBeenCalledTimes(1);

		await retryMediaToolCatalog();
		await retryMediaToolCatalog();

		expect(listMediaTools).toHaveBeenCalledTimes(2);
		expect(toastShow).toHaveBeenCalledTimes(1);
	});
});
