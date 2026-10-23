import { writable, type Readable } from 'svelte/store';
import { CanceledError } from 'axios';
import { api } from '$lib/services/api/index';

export type TagType = 'GENERATION' | 'MODEL' | 'UPLOAD';

export const TAG_CATALOG_TTL_MS = 15_000;

export interface CatalogTag {
	id: string;
	name: string;
	type?: string;
	usage_count?: number;
	model_count?: number;
	[key: string]: unknown;
}

interface GetOptions {
	signal?: AbortSignal;
	fresh?: boolean;
}

interface Entry {
	store: ReturnType<typeof writable<CatalogTag[]>>;
	tags: CatalogTag[] | null;
	fetchedAt: number;
	epoch: number;
	inFlight: Promise<CatalogTag[]> | null;
	controller: AbortController | null;
	waiters: number;
}

function createTagCatalog() {
	const entries = new Map<TagType, Entry>();

	function entryFor(type: TagType): Entry {
		let entry = entries.get(type);
		if (!entry) {
			entry = {
				store: writable<CatalogTag[]>([]),
				tags: null,
				fetchedAt: 0,
				epoch: 0,
				inFlight: null,
				controller: null,
				waiters: 0
			};
			entries.set(type, entry);
		}
		return entry;
	}

	function launch(type: TagType, entry: Entry): Promise<CatalogTag[]> {
		const controller = new AbortController();
		const epoch = entry.epoch;
		entry.controller = controller;
		const request: Promise<CatalogTag[]> = Promise.resolve(api.getTags(type, controller.signal))
			.then((response) => {
				if (!response?.success || !response.data?.tags) return entry.tags ?? [];
				const tags = response.data.tags as CatalogTag[];
				if (epoch === entry.epoch) {
					entry.tags = tags;
					entry.fetchedAt = Date.now();
					entry.store.set(tags);
				}
				return tags;
			})
			.finally(() => {
				if (entry.inFlight === request) {
					entry.inFlight = null;
					entry.controller = null;
				}
			});
		entry.inFlight = request;
		return request;
	}

	function abandon(entry: Entry) {
		entry.waiters--;
		if (entry.waiters > 0) return;
		entry.controller?.abort();
		entry.controller = null;
		entry.inFlight = null;
	}

	function waitFor(entry: Entry, request: Promise<CatalogTag[]>, signal?: AbortSignal) {
		entry.waiters++;
		if (!signal) return request.finally(() => entry.waiters--);
		return new Promise<CatalogTag[]>((resolve, reject) => {
			let settled = false;
			const onAbort = () => {
				if (settled) return;
				settled = true;
				abandon(entry);
				reject(new CanceledError());
			};
			if (signal.aborted) return onAbort();
			signal.addEventListener('abort', onAbort, { once: true });
			request.then(
				(tags) => {
					if (settled) return;
					settled = true;
					entry.waiters--;
					signal.removeEventListener('abort', onAbort);
					resolve(tags);
				},
				(error) => {
					if (settled) return;
					settled = true;
					entry.waiters--;
					signal.removeEventListener('abort', onAbort);
					reject(error);
				}
			);
		});
	}

	function invalidate(type: TagType) {
		const entry = entryFor(type);
		entry.epoch++;
		entry.fetchedAt = 0;
		entry.inFlight = null;
	}

	return {
		tags(type: TagType): Readable<CatalogTag[]> {
			return { subscribe: entryFor(type).store.subscribe };
		},

		async get<T = CatalogTag>(type: TagType, options: GetOptions = {}): Promise<T[]> {
			if (options.signal?.aborted) throw new CanceledError();
			const entry = entryFor(type);
			const usable =
				entry.tags !== null && !options.fresh && Date.now() - entry.fetchedAt < TAG_CATALOG_TTL_MS;
			if (usable) return entry.tags as unknown as T[];
			const request = entry.inFlight ?? launch(type, entry);
			return (await waitFor(entry, request, options.signal)) as unknown as T[];
		},

		async create(name: string, type: TagType) {
			const response = await api.createTag(name, type);
			if (response.success) {
				const created = response.data?.tag as CatalogTag | undefined;
				const entry = entryFor(type);
				if (created && !entry.tags?.some((tag) => tag.id === created.id)) {
					entry.tags = [...(entry.tags ?? []), created];
					entry.store.set(entry.tags);
				}
				invalidate(type);
			}
			return response;
		},

		async remove(tagId: string, type: TagType) {
			const response = await api.deleteTag(tagId);
			if (response.success) {
				const entry = entryFor(type);
				if (entry.tags) {
					entry.tags = entry.tags.filter((tag) => tag.id !== tagId);
					entry.store.set(entry.tags);
				}
				invalidate(type);
			}
			return response;
		},

		invalidate,

		reset() {
			for (const entry of entries.values()) {
				entry.controller?.abort();
				entry.epoch++;
				entry.tags = null;
				entry.fetchedAt = 0;
				entry.inFlight = null;
				entry.controller = null;
				entry.waiters = 0;
				entry.store.set([]);
			}
		}
	};
}

export const tagCatalog = createTagCatalog();
