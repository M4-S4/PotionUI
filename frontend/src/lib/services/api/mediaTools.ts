import type { AxiosInstance } from 'axios';
import { api } from '$lib/services/api/index';
import type { APIResponse } from '$lib/types/api';
import type {
	MediaToolRun,
	PreviewMediaToolBody,
	StartMediaToolRunBody,
	ToolDTO,
	ToolMediaKind,
	ToolScope
} from '$lib/media-tools/types';

function unwrap<T>(body: APIResponse<T> | undefined, fallback: string): T {
	if (!body || !body.success || body.data === undefined) {
		throw new Error(body?.message || body?.error || fallback);
	}
	return body.data;
}

export function createMediaToolsApi(client: AxiosInstance) {
	return {
		async listMediaTools(kind?: ToolMediaKind, scope?: ToolScope): Promise<ToolDTO[]> {
			const params: Record<string, string> = {};
			if (kind) params.kind = kind;
			if (scope) params.scope = scope;
			const response = await client.get<APIResponse<{ tools: ToolDTO[] }>>('/api/media-tools', {
				params
			});
			return unwrap(response.data, 'The tool list could not be loaded').tools ?? [];
		},

		async startMediaToolRun(body: StartMediaToolRunBody): Promise<{ run_id: string }> {
			const response = await client.post<APIResponse<{ run_id: string }>>(
				'/api/media-tools/runs',
				body
			);
			return unwrap(response.data, 'The edit could not be started');
		},

		async getMediaToolRun(runId: string, signal?: AbortSignal): Promise<MediaToolRun> {
			const response = await client.get<APIResponse<MediaToolRun>>(
				`/api/media-tools/runs/${encodeURIComponent(runId)}`,
				{ signal }
			);
			return unwrap(response.data, 'The edit status could not be read');
		},

		async cancelMediaToolRun(runId: string): Promise<void> {
			await client.delete(`/api/media-tools/runs/${encodeURIComponent(runId)}`);
		},

		async previewMediaTool(body: PreviewMediaToolBody, signal?: AbortSignal): Promise<Blob> {
			const response = await client.post<Blob>('/api/media-tools/preview', body, {
				responseType: 'blob',
				signal
			});
			return response.data;
		}
	};
}

export type MediaToolsApi = ReturnType<typeof createMediaToolsApi>;

function bound(): MediaToolsApi {
	return createMediaToolsApi(api.getClient());
}

export async function listMediaTools(kind?: ToolMediaKind, scope?: ToolScope): Promise<ToolDTO[]> {
	return bound().listMediaTools(kind, scope);
}

export async function startMediaToolRun(body: StartMediaToolRunBody): Promise<{ run_id: string }> {
	return bound().startMediaToolRun(body);
}

export async function getMediaToolRun(runId: string, signal?: AbortSignal): Promise<MediaToolRun> {
	return bound().getMediaToolRun(runId, signal);
}

export async function cancelMediaToolRun(runId: string): Promise<void> {
	return bound().cancelMediaToolRun(runId);
}

export async function previewMediaTool(
	body: PreviewMediaToolBody,
	signal?: AbortSignal
): Promise<Blob> {
	return bound().previewMediaTool(body, signal);
}
