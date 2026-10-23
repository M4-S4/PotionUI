import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('$lib/plugin-api/componentResolver', () => ({
	resolvePluginComponent: vi.fn(),
	getPluginRevision: () => 'rev-1'
}));

import { resolvePluginComponent } from '$lib/plugin-api/componentResolver';
import { chatToolRendererRegistry } from './chatToolRendererRegistry';
import { artifactRendererRegistry } from './artifactRendererRegistry';

describe.each([
	{ name: 'chatToolRendererRegistry', registry: chatToolRendererRegistry, prefix: 'chat_tool' },
	{ name: 'artifactRendererRegistry', registry: artifactRendererRegistry, prefix: 'artifact' }
])('$name', ({ registry, prefix }) => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	it('resolves a statically registered core component', async () => {
		const Component = { name: 'CoreComponent' };
		registry.register(`${prefix}_static`, { component: Component });

		expect(registry.has(`${prefix}_static`)).toBe(true);
		await expect(registry.resolve(`${prefix}_static`)).resolves.toBe(Component);
		expect(resolvePluginComponent).not.toHaveBeenCalled();
	});

	it('returns null for an unregistered key', async () => {
		await expect(registry.resolve(`${prefix}_unknown`)).resolves.toBeNull();
	});

	it('resolves a lazy plugin entry via componentResolver and caches the result', async () => {
		const PluginComponent = { name: 'PluginComponent' };
		vi.mocked(resolvePluginComponent).mockResolvedValue(PluginComponent);

		registry.register(`${prefix}_lazy`, { pluginId: 'example-extensions', asset: 'Renderer.js' });

		const first = await registry.resolve(`${prefix}_lazy`);
		const second = await registry.resolve(`${prefix}_lazy`);

		expect(first).toBe(PluginComponent);
		expect(second).toBe(PluginComponent);
		expect(resolvePluginComponent).toHaveBeenCalledTimes(1);
		expect(resolvePluginComponent).toHaveBeenCalledWith('example-extensions', 'Renderer.js');
	});

	it('unregister removes the entry and clears the cache', async () => {
		registry.register(`${prefix}_scratch`, { component: {} });
		expect(registry.has(`${prefix}_scratch`)).toBe(true);

		registry.unregister(`${prefix}_scratch`);

		expect(registry.has(`${prefix}_scratch`)).toBe(false);
		await expect(registry.resolve(`${prefix}_scratch`)).resolves.toBeNull();
	});

	it('keys reflects registered entries', () => {
		registry.register(`${prefix}_listed`, { component: {} });
		expect(registry.keys()).toContain(`${prefix}_listed`);
	});
});
