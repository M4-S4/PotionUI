import type { PluginHostFetchApi } from './hostApi';

export function createPluginPageApi(hostApi: PluginHostFetchApi, getToken: () => string | null, pluginId: string) {
	const prefix = `/api/plugins/${pluginId}`;

	return {
		getToken,
		get: (path: string) => hostApi.json(`${prefix}${path}`),
		post: (path: string, data: any) =>
			hostApi.json(`${prefix}${path}`, { method: 'POST', body: JSON.stringify(data) }),
		put: (path: string, data: any) =>
			hostApi.json(`${prefix}${path}`, { method: 'PUT', body: JSON.stringify(data) }),
		delete: (path: string) => hostApi.json(`${prefix}${path}`, { method: 'DELETE' }),
		uploadFile: (path: string, formData: FormData) =>
			hostApi.json(`${prefix}${path}`, { method: 'POST', body: formData, timeout: 0 }),
		fetchFromApp: (path: string) => hostApi.json(path)
	};
}
