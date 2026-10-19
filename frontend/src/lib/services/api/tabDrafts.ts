import type { AxiosInstance } from 'axios';
import type { APIResponse } from '$lib/types/api';
import type { VariablesMap } from '$lib/utils/variableDefs';

export interface TabDraftSummary {
	tab_id: string;
	variable_count: number;
	rev: number;
	updated_at: string | null;
}

export interface TabDraftDoc {
	variables?: VariablesMap;
}

export interface TabDraft {
	tab_id: string;
	doc: TabDraftDoc;
	rev: number;
	created_at: string | null;
	updated_at: string | null;
}

export interface TabDraftOp {
	op: 'set' | 'remove';
	name: string;
	type?: 'text' | 'choice' | 'number';
	value?: string | number;
	options?: unknown[];
	mode?: string;
	pinned_index?: number | null;
	min?: number;
	max?: number;
	step?: number;
}

export interface TabDraftOpsRequest {
	ops: TabDraftOp[];
	base_rev?: number;
}

export function createTabDraftsApi(client: AxiosInstance) {
	const path = (tabId: string) => `/api/tab-drafts/${encodeURIComponent(tabId)}`;

	return {
		async listTabDrafts(): Promise<APIResponse<TabDraftSummary[]>> {
			const response = await client.get('/api/tab-drafts');
			return response.data;
		},

		async getTabDraft(tabId: string): Promise<APIResponse<TabDraft>> {
			const response = await client.get(path(tabId));
			return response.data;
		},

		async replaceTabDraft(tabId: string, doc: TabDraftDoc): Promise<APIResponse<TabDraft>> {
			const response = await client.put(path(tabId), { doc });
			return response.data;
		},

		async applyTabDraftOps(tabId: string, request: TabDraftOpsRequest): Promise<APIResponse<TabDraft>> {
			const response = await client.post(`${path(tabId)}/ops`, request);
			return response.data;
		},

		async deleteTabDraft(tabId: string): Promise<void> {
			await client.delete(path(tabId));
		}
	};
}
