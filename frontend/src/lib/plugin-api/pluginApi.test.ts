// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createPluginApi, PluginApiError, type PluginApiOptions } from './pluginApi';

type Call = { url: string; init: RequestInit };

function setup(respond: (call: Call) => Response, token: string | null = 'tok') {
	const calls: Call[] = [];
	const fetch = vi.fn(async (url: string, init: RequestInit = {}) => {
		const call = { url, init };
		calls.push(call);
		return respond(call);
	});
	const options: PluginApiOptions = { baseUrl: 'http://host', getToken: () => token, fetch };
	return { api: createPluginApi('video-producer', options), calls };
}

function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
	return new Response(JSON.stringify(body), {
		status,
		headers: { 'Content-Type': 'application/json', ...headers }
	});
}

function headerOf(init: RequestInit, name: string): string | null {
	return new Headers(init.headers).get(name);
}

describe('createPluginApi: the methods model-browser relies on', () => {
	it('sends JSON with the bearer token to the plugin routes', async () => {
		const { api, calls } = setup(() => json({ success: true, data: 1 }));

		await api.get('/projects');
		await api.post('/projects', { name: 'Teaser' });
		await api.put('/projects/1', { name: 'Trailer' });
		await api.delete('/projects/1');

		expect(calls.map((call) => [call.init.method ?? 'GET', call.url])).toEqual([
			['GET', 'http://host/api/plugins/video-producer/projects'],
			['POST', 'http://host/api/plugins/video-producer/projects'],
			['PUT', 'http://host/api/plugins/video-producer/projects/1'],
			['DELETE', 'http://host/api/plugins/video-producer/projects/1']
		]);
		for (const { init } of calls) {
			expect(init.credentials).toBe('include');
			expect(init.headers).toEqual({ 'Content-Type': 'application/json', Authorization: 'Bearer tok' });
		}
		expect(calls[1].init.body).toBe('{"name":"Teaser"}');
	});

	it('still resolves with the parsed body of an error response', async () => {
		const { api } = setup(() => json({ detail: { success: false, error: 'not_found', message: 'Gone' } }, 404));

		await expect(api.get('/models/1')).resolves.toEqual({
			detail: { success: false, error: 'not_found', message: 'Gone' }
		});
	});

	it('still rejects with a SyntaxError when the body is not JSON', async () => {
		const { api } = setup(() => new Response('<html>Bad gateway</html>', { status: 502 }));

		await expect(api.get('/models')).rejects.toBeInstanceOf(SyntaxError);
	});

	it('uploads form data without a content type and reads core routes through fetchFromApp', async () => {
		const { api, calls } = setup(() => json({ success: true }));
		const form = new FormData();

		await api.uploadFile('/media', form);
		await api.fetchFromApp('/api/backends');

		expect(calls[0].init).toMatchObject({ method: 'POST', body: form, headers: { Authorization: 'Bearer tok' } });
		expect(calls[1].url).toBe('http://host/api/backends');
	});

	it('exposes the token and omits the header when there is none', async () => {
		const { api, calls } = setup(() => json({}), null);

		expect(api.getToken()).toBeNull();
		await api.get('/me');
		expect(calls[0].init.headers).toEqual({ 'Content-Type': 'application/json' });
	});
});

describe('createPluginApi: request and patch', () => {
	it('returns the parsed JSON body of a success', async () => {
		const { api, calls } = setup(() => json({ success: true, data: { id: 3 } }));

		await expect(api.request('get', '/shots/3')).resolves.toEqual({ success: true, data: { id: 3 } });
		expect(calls[0].init.method).toBe('GET');
		expect(calls[0].init.body).toBeUndefined();
		expect(headerOf(calls[0].init, 'Content-Type')).toBeNull();
		expect(headerOf(calls[0].init, 'Authorization')).toBe('Bearer tok');
	});

	it('patches with a JSON body', async () => {
		const { api, calls } = setup(() => json({ success: true }));

		await api.patch('/shots/3', { duration: 4 });

		expect(calls[0].url).toBe('http://host/api/plugins/video-producer/shots/3');
		expect(calls[0].init.method).toBe('PATCH');
		expect(calls[0].init.body).toBe('{"duration":4}');
		expect(headerOf(calls[0].init, 'Content-Type')).toBe('application/json');
		expect(calls[0].init.credentials).toBe('include');
	});

	it('returns null for 204 No Content', async () => {
		const { api } = setup(() => new Response(null, { status: 204 }));

		await expect(api.request('DELETE', '/shots/3')).resolves.toBeNull();
	});

	it('throws the status, the body and the detail of a 4xx', async () => {
		const body = { detail: 'Shot 3 is locked' };
		const { api } = setup(() => json(body, 409));

		const error = await api.patch('/shots/3', {}).catch((err) => err);

		expect(error).toBeInstanceOf(PluginApiError);
		expect(error).toMatchObject({ status: 409, body, detail: 'Shot 3 is locked', message: 'Shot 3 is locked' });
	});

	it('takes the message of a structured detail', async () => {
		const detail = { error: 'limit_exceeded', message: 'Quota reached' };
		const { api } = setup(() => json({ detail }, 403));

		await expect(api.request('POST', '/renders', {})).rejects.toMatchObject({
			status: 403,
			detail,
			message: 'Quota reached'
		});
	});

	it('throws the status and the raw text of a 5xx that is not JSON', async () => {
		const { api } = setup(() => new Response('<html>Bad gateway</html>', { status: 502 }));

		await expect(api.request('GET', '/renders')).rejects.toMatchObject({
			status: 502,
			body: '<html>Bad gateway</html>',
			detail: undefined,
			message: 'Request failed with status 502'
		});
	});

	it('lets a network failure through as it is', async () => {
		const { api } = setup(() => {
			throw new TypeError('Failed to fetch');
		});

		await expect(api.request('GET', '/renders')).rejects.toBeInstanceOf(TypeError);
	});
});

describe('createPluginApi: fetchRaw', () => {
	it('returns the response of a binary route with the bearer set', async () => {
		const bytes = new Uint8Array([0, 159, 146, 150]);
		const { api, calls } = setup(() => new Response(bytes, { status: 200, headers: { 'Content-Type': 'video/mp4' } }));

		const res = await api.fetchRaw('/renders/1/file', { headers: { Range: 'bytes=0-3' } });

		expect(new Uint8Array(await res.arrayBuffer())).toEqual(bytes);
		expect(calls[0].url).toBe('http://host/api/plugins/video-producer/renders/1/file');
		expect(calls[0].init.credentials).toBe('include');
		expect(headerOf(calls[0].init, 'Authorization')).toBe('Bearer tok');
		expect(headerOf(calls[0].init, 'Range')).toBe('bytes=0-3');
	});

	it('hands back an error response instead of throwing', async () => {
		const { api } = setup(() => json({ detail: 'Not found' }, 404));

		const res = await api.fetchRaw('/renders/9/file');

		expect(res.status).toBe(404);
	});
});

describe('createPluginApi: download', () => {
	let clicked: HTMLAnchorElement[];

	beforeEach(() => {
		vi.useFakeTimers();
		clicked = [];
		Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:export-1') });
		Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
		vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
			clicked.push(this);
		});
	});

	afterEach(() => {
		vi.useRealTimers();
		vi.restoreAllMocks();
	});

	it('saves the response as a file, then removes the link and revokes the object URL', async () => {
		const { api, calls } = setup(() => new Response('<fcpxml/>', { status: 200 }));

		await api.download('/projects/1/export.fcpxml', 'teaser.fcpxml');

		expect(headerOf(calls[0].init, 'Authorization')).toBe('Bearer tok');
		expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
		expect(clicked).toHaveLength(1);
		expect(clicked[0].download).toBe('teaser.fcpxml');
		expect(clicked[0].href).toBe('blob:export-1');
		expect(clicked[0].isConnected).toBe(false);
		expect(URL.revokeObjectURL).not.toHaveBeenCalled();
		vi.advanceTimersByTime(59_999);
		expect(URL.revokeObjectURL).not.toHaveBeenCalled();
		vi.advanceTimersByTime(1);
		expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:export-1');
	});

	it('names the file after Content-Disposition, then after the path', async () => {
		const { api } = setup(({ url }) =>
			url.endsWith('/srt')
				? new Response('1', { headers: { 'Content-Disposition': "attachment; filename*=UTF-8''sous-titres%20FR.srt" } })
				: new Response('1')
		);

		await api.download('/projects/1/srt');
		await api.download('/projects/1/edl?fps=25');

		expect(clicked.map((link) => link.download)).toEqual(['sous-titres FR.srt', 'edl']);
	});

	it('throws on an error response without creating a file', async () => {
		const { api } = setup(() => json({ detail: 'Export not ready' }, 409));

		await expect(api.download('/projects/1/export.otio', 'cut.otio')).rejects.toMatchObject({
			status: 409,
			detail: 'Export not ready'
		});
		expect(URL.createObjectURL).not.toHaveBeenCalled();
		expect(clicked).toEqual([]);
	});
});
