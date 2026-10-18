export interface PluginPageApi {
	getToken(): string | null;
	get(path: string): Promise<any>;
	post(path: string, data: unknown): Promise<any>;
	put(path: string, data: unknown): Promise<any>;
	delete(path: string): Promise<any>;
	uploadFile(path: string, formData: FormData): Promise<any>;
	fetchFromApp(path: string): Promise<any>;
	request(method: string, path: string, data?: unknown): Promise<any>;
	patch(path: string, data: unknown): Promise<any>;
	fetchRaw(path: string, init?: RequestInit): Promise<Response>;
	download(path: string, filename?: string): Promise<void>;
}

export interface PluginApiOptions {
	baseUrl: string;
	getToken: () => string | null;
	fetch?: (input: string, init?: RequestInit) => Promise<Response>;
}

export class PluginApiError extends Error {
	readonly status: number;
	readonly body: unknown;
	readonly detail: unknown;

	constructor(status: number, body: unknown) {
		const detail = isRecord(body) && 'detail' in body ? body.detail : undefined;
		super(errorMessage(status, detail));
		this.name = 'PluginApiError';
		this.status = status;
		this.body = body;
		this.detail = detail;
	}
}

const DOWNLOAD_URL_LIFETIME_MS = 60_000;

function isRecord(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function errorMessage(status: number, detail: unknown): string {
	if (typeof detail === 'string' && detail) return detail;
	if (isRecord(detail) && typeof detail.message === 'string' && detail.message) return detail.message;
	return `Request failed with status ${status}`;
}

async function readBody(res: Response): Promise<unknown> {
	if (res.status === 204) return null;
	const text = await res.text();
	if (!text) return null;
	try {
		return JSON.parse(text);
	} catch {
		return text;
	}
}

function filenameFrom(res: Response, path: string): string {
	const disposition = res.headers.get('Content-Disposition') ?? '';
	const encoded = disposition.match(/filename\*\s*=\s*(?:UTF-8|utf-8)''([^;]+)/);
	if (encoded) {
		try {
			return decodeURIComponent(encoded[1].trim());
		} catch {
			return encoded[1].trim();
		}
	}
	const plain = disposition.match(/filename\s*=\s*"?([^";]+)"?/);
	if (plain) return plain[1].trim();
	return path.split('?')[0].split('/').filter(Boolean).pop() || 'download';
}

export function createPluginApi(pluginId: string, options: PluginApiOptions): PluginPageApi {
	const fetchImpl = options.fetch ?? ((input: string, init?: RequestInit) => fetch(input, init));
	const pluginUrl = (path: string) => `${options.baseUrl}/api/plugins/${pluginId}${path}`;
	const bearer = (): Record<string, string> => {
		const token = options.getToken();
		return token ? { Authorization: `Bearer ${token}` } : {};
	};
	const headers = () => ({ 'Content-Type': 'application/json', ...bearer() });

	const fetchRaw = (path: string, init: RequestInit = {}) => {
		const merged = new Headers(init.headers);
		for (const [name, value] of Object.entries(bearer())) merged.set(name, value);
		return fetchImpl(pluginUrl(path), { credentials: 'include', ...init, headers: merged });
	};

	const request = async (method: string, path: string, data?: unknown) => {
		const init: RequestInit = { method: method.toUpperCase(), headers: bearer() };
		if (data !== undefined) {
			init.headers = headers();
			init.body = JSON.stringify(data);
		}
		const res = await fetchRaw(path, init);
		const body = await readBody(res);
		if (!res.ok) throw new PluginApiError(res.status, body);
		return body;
	};

	return {
		getToken: () => options.getToken(),
		get: async (path) => {
			const res = await fetchImpl(pluginUrl(path), {
				headers: headers(),
				credentials: 'include'
			});
			return res.json();
		},
		post: async (path, data) => {
			const res = await fetchImpl(pluginUrl(path), {
				method: 'POST',
				headers: headers(),
				credentials: 'include',
				body: JSON.stringify(data)
			});
			return res.json();
		},
		put: async (path, data) => {
			const res = await fetchImpl(pluginUrl(path), {
				method: 'PUT',
				headers: headers(),
				credentials: 'include',
				body: JSON.stringify(data)
			});
			return res.json();
		},
		delete: async (path) => {
			const res = await fetchImpl(pluginUrl(path), {
				method: 'DELETE',
				headers: headers(),
				credentials: 'include'
			});
			return res.json();
		},
		uploadFile: async (path, formData) => {
			const res = await fetchImpl(pluginUrl(path), {
				method: 'POST',
				headers: bearer(),
				credentials: 'include',
				body: formData
			});
			return res.json();
		},
		fetchFromApp: async (path) => {
			const res = await fetchImpl(`${options.baseUrl}${path}`, {
				headers: headers(),
				credentials: 'include'
			});
			return res.json();
		},
		request,
		patch: (path, data) => request('PATCH', path, data),
		fetchRaw,
		download: async (path, filename) => {
			const res = await fetchRaw(path);
			if (!res.ok) throw new PluginApiError(res.status, await readBody(res));
			const url = URL.createObjectURL(await res.blob());
			const link = document.createElement('a');
			link.href = url;
			link.download = filename || filenameFrom(res, path);
			link.style.display = 'none';
			document.body.appendChild(link);
			try {
				link.click();
			} finally {
				link.remove();
				setTimeout(() => URL.revokeObjectURL(url), DOWNLOAD_URL_LIFETIME_MS);
			}
		}
	};
}
