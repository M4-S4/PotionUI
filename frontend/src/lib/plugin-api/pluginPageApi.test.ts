import { describe, it, expect, vi } from 'vitest';
import { createPluginPageApi } from './pluginPageApi';
import { createHostFetchApi } from './hostApi';

function setup() {
	const host = { fetch: vi.fn(), json: vi.fn(async () => ({ ok: true })) };
	const api = createPluginPageApi(host, () => 'tok', 'demo');
	return { host, api };
}

describe('plugin page api', () => {
	it('prefixes get with the plugin route', async () => {
		const { host, api } = setup();
		expect(await api.get('/items?x=1')).toEqual({ ok: true });
		expect(host.json).toHaveBeenCalledWith('/api/plugins/demo/items?x=1');
	});

	it.each([
		['post', 'POST'],
		['put', 'PUT']
	] as const)('%s sends a JSON body', async (name, method) => {
		const { host, api } = setup();
		await api[name]('/items', { a: 1 });
		expect(host.json).toHaveBeenCalledWith('/api/plugins/demo/items', { method, body: '{"a":1}' });
	});

	it('delete uses DELETE without a body', async () => {
		const { host, api } = setup();
		await api.delete('/items/3');
		expect(host.json).toHaveBeenCalledWith('/api/plugins/demo/items/3', { method: 'DELETE' });
	});

	it('uploadFile passes the FormData through with no timeout', async () => {
		const { host, api } = setup();
		const form = new FormData();
		await api.uploadFile('/upload', form);
		const [url, init] = host.json.mock.calls[0] as unknown as [string, any];
		expect(url).toBe('/api/plugins/demo/upload');
		expect(init).toMatchObject({ method: 'POST', timeout: 0 });
		expect(init.body).toBe(form);
	});

	it('fetchFromApp keeps the path unprefixed and exposes the token', async () => {
		const { host, api } = setup();
		await api.fetchFromApp('/api/models');
		expect(host.json).toHaveBeenCalledWith('/api/models');
		expect(api.getToken()).toBe('tok');
	});

	it('fetchFromApp cannot reach another host', async () => {
		const request = vi.fn();
		const client = { getClient: () => ({ request }) as any, triggerAuthExpired: vi.fn() };
		const api = createPluginPageApi(createHostFetchApi(client), () => null, 'demo');
		await expect(api.fetchFromApp(`https:${'/'.repeat(2)}evil.example/x`)).rejects.toBeInstanceOf(TypeError);
		expect(request).not.toHaveBeenCalled();
	});
});
