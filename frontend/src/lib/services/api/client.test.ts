import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { InternalAxiosRequestConfig } from 'axios';
import { APIClient } from './client';

vi.mock('$lib/utils/storage', () => ({
	storage: { get: () => null, set: () => undefined, remove: () => undefined }
}));

function clientCapturing(seen: InternalAxiosRequestConfig[]) {
	const api = new APIClient('');
	api.getClient().defaults.adapter = async (config) => {
		seen.push(config);
		return { data: { success: true }, status: 200, statusText: 'OK', headers: {}, config };
	};
	return api;
}

describe('APIClient request bodies', () => {
	let seen: InternalAxiosRequestConfig[];
	beforeEach(() => {
		seen = [];
	});

	it('sends a FormData body as multipart, never re-serialised to JSON', async () => {
		const api = clientCapturing(seen);
		const formData = new FormData();
		formData.append('file', new Blob(['png-bytes'], { type: 'image/png' }), 'pasted.png');

		await api.getClient().post('/api/media/upload', formData);

		expect(seen).toHaveLength(1);
		expect(seen[0].data).toBeInstanceOf(FormData);
		expect(String(seen[0].headers['Content-Type'])).toContain('multipart/form-data');
	});

	it('keeps JSON for plain object bodies', async () => {
		const api = clientCapturing(seen);

		await api.getClient().post('/api/things', { a: 1 });

		expect(seen[0].data).toBe(JSON.stringify({ a: 1 }));
		expect(String(seen[0].headers['Content-Type'])).toContain('application/json');
	});
});

describe('APIClient token renewal', () => {
	function clientRenewing(seen: InternalAxiosRequestConfig[], renewed: string | null, beforeReply?: () => void) {
		const api = new APIClient('');
		let pending = renewed;
		api.getClient().defaults.adapter = async (config) => {
			seen.push(config);
			beforeReply?.();
			const headers = pending ? { 'x-refreshed-token': pending } : {};
			pending = null;
			return { data: { success: true }, status: 200, statusText: 'OK', headers, config };
		};
		return api;
	}

	it('adopts a renewed token delivered for the current bearer and sends it from then on', async () => {
		const seen: InternalAxiosRequestConfig[] = [];
		const api = clientRenewing(seen, 'renewed-token');
		const onRefreshed = vi.fn();
		api.setOnTokenRefreshed(onRefreshed);
		api.setAuthHeader('original-token');

		await api.getClient().get('/api/things');
		await api.getClient().get('/api/things');

		expect(api.getToken()).toBe('renewed-token');
		expect(onRefreshed).toHaveBeenCalledTimes(1);
		expect(onRefreshed).toHaveBeenCalledWith('original-token', 'renewed-token');
		expect(String(seen[0].headers.Authorization)).toBe('Bearer original-token');
		expect(String(seen[1].headers.Authorization)).toBe('Bearer renewed-token');
	});

	it('ignores a renewed token when the session changed while the request was in flight', async () => {
		const seen: InternalAxiosRequestConfig[] = [];
		let api!: APIClient;
		api = clientRenewing(seen, 'renewed-for-old-session', () => api.setAuthHeader('other-session-token'));
		const onRefreshed = vi.fn();
		api.setOnTokenRefreshed(onRefreshed);
		api.setAuthHeader('original-token');

		await api.getClient().get('/api/things');

		expect(api.getToken()).toBe('other-session-token');
		expect(onRefreshed).not.toHaveBeenCalled();
	});

	it('ignores a renewed token on a request that carried no bearer', async () => {
		const seen: InternalAxiosRequestConfig[] = [];
		const api = clientRenewing(seen, 'renewed-token');
		const onRefreshed = vi.fn();
		api.setOnTokenRefreshed(onRefreshed);

		await api.getClient().get('/api/things');

		expect(api.getToken()).toBeNull();
		expect(onRefreshed).not.toHaveBeenCalled();
	});

	it('keeps the token when the response carries no renewal', async () => {
		const seen: InternalAxiosRequestConfig[] = [];
		const api = clientRenewing(seen, null);
		const onRefreshed = vi.fn();
		api.setOnTokenRefreshed(onRefreshed);
		api.setAuthHeader('original-token');

		await api.getClient().get('/api/things');

		expect(api.getToken()).toBe('original-token');
		expect(onRefreshed).not.toHaveBeenCalled();
	});
});

