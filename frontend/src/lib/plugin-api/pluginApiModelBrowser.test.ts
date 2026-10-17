import { describe, it, expect } from 'vitest';
import { resolve } from 'node:path';
import { createPluginApi } from './pluginApi';

const MODEL_BROWSER_CLIENT = resolve(
	__dirname,
	'../../../../content/plugins/marketplace/model-browser/frontend/src/lib/client.js'
);

function json(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

function route(url: string): Response {
	if (url.endsWith('/me')) return json({ success: true, data: { id: 'u1' } });
	if (url.includes('/search')) {
		return json({ detail: { success: false, error: 'rate_limited', message: 'try again in 30 s' } }, 429);
	}
	if (url.includes('/models/')) return json({ detail: { success: false, error: 'not_found', message: 'Gone' } }, 404);
	return new Response('<html>oops</html>', { status: 500 });
}

describe('model-browser client on the extracted plugin api', () => {
	it('reads success, error envelopes, rate limits and broken bodies as before', async () => {
		const { getMe, getModel, searchModels, callPlugin } = await import(/* @vite-ignore */ MODEL_BROWSER_CLIENT);
		const api = createPluginApi('model-browser', {
			baseUrl: '',
			getToken: () => 'tok',
			fetch: async (url) => route(url)
		});

		expect(await getMe(api)).toEqual({ ok: true, data: { id: 'u1' } });
		expect(await getModel(api, 'civitai', '1')).toEqual({
			ok: false,
			error: { status: 404, code: 'not_found', message: 'Gone', retryAfter: null }
		});
		expect((await searchModels(api, 'civitai', '')).error).toEqual({
			status: 429,
			code: 'rate_limited',
			message: 'try again in 30 s',
			retryAfter: 30
		});
		expect((await callPlugin(api, 'get', '/broken')).error.code).toBe('invalid_response');
	});
});
