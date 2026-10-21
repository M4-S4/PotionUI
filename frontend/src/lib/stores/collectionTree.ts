import { writable, type Readable } from 'svelte/store';
import { logger, getErrorMessage } from '$lib/utils/logger';
import type { SmartCounts } from '$lib/components/collections/types';

export interface CollectionTreeState<T> {
	collections: T[];
	smartCounts: SmartCounts | null;
	loading: boolean;
}

interface Result {
	success: boolean;
	error?: string;
	message?: string;
	data?: unknown;
}

interface BulkMoveResponse {
	success: boolean;
	error?: string;
	message?: string;
	data?: { moved?: number; failed?: number; errors?: { id: string; reason: string }[] };
}

export interface BulkMoveOutcome {
	success: boolean;
	error?: string;
	message?: string;
	moved: number;
	failed: number;
	errors: { id: string; reason: string }[];
}

export interface CollectionTreeAdapter<T extends { id: string }> {
	list(directOnly: boolean): Promise<{ collections: T[]; smartCounts: SmartCounts | null } | null>;
	create(name: string, parentId?: string | null): Promise<Result & { data?: { collection?: T } }>;
	rename(id: string, name: string): Promise<Result>;
	move(id: string, parentId: string | null): Promise<Result>;
	remove(id: string): Promise<Result>;
	addMembers(id: string, memberIds: string[]): Promise<Result>;
	removeMembers?(id: string, memberIds: string[]): Promise<Result>;
	bulkMove?(ids: string[], parentId: string | null): Promise<BulkMoveResponse>;
}

interface CollectionTreeOptions {
	label: string;
	cached?: boolean;
}

type ItemOf<A> = A extends { list(directOnly: boolean): Promise<{ collections: (infer T)[] } | null> }
	? T
	: never;

type CollectionTreeStore<A extends CollectionTreeAdapter<any>> = Readable<CollectionTreeState<ItemOf<A>>> & {
	load(force?: boolean): Promise<void>;
	setDirectOnly(value: boolean): Promise<void>;
	create: A['create'];
	rename: A['rename'];
	move: A['move'];
	remove: A['remove'];
	addMembers: A['addMembers'];
	reset(): void;
} & (A extends { removeMembers: unknown } ? { removeMembers: A['addMembers'] } : unknown) &
	(A extends { bulkMove: unknown }
		? { bulkMove(ids: string[], parentId: string | null): Promise<BulkMoveOutcome> }
		: unknown);

export function createCollectionTreeStore<A extends CollectionTreeAdapter<any>>(
	adapter: A,
	{ label, cached = false }: CollectionTreeOptions
): CollectionTreeStore<A> {
	type T = ItemOf<A>;
	const initialState: CollectionTreeState<T> = { collections: [], smartCounts: null, loading: false };
	const { subscribe, update, set } = writable<CollectionTreeState<T>>(initialState);

	let requestSeq = 0;
	let lifetime = 0;
	let directOnly = false;
	let loaded = false;
	let inFlight: Promise<void> | null = null;
	let forcedAfterFlight: Promise<void> | null = null;

	async function fetchCollections() {
		const seq = ++requestSeq;
		const requestLifetime = lifetime;
		const isStale = () => requestLifetime !== lifetime || seq !== requestSeq;
		update((state) => ({ ...state, loading: true }));
		try {
			const result = await adapter.list(directOnly);
			if (isStale()) return;
			if (result) {
				update((state) => ({
					...state,
					collections: result.collections,
					smartCounts: result.smartCounts,
					loading: false
				}));
				loaded = true;
			} else {
				update((state) => ({ ...state, loading: false }));
			}
		} catch (error) {
			logger.error(`Failed to load ${label}:`, getErrorMessage(error));
			if (isStale()) return;
			update((state) => ({ ...state, loading: false }));
		}
	}

	function refresh(force = false): Promise<void> {
		if (!cached) return fetchCollections();
		if (!force && loaded) return Promise.resolve();
		if (inFlight) {
			if (!force) return inFlight;
			if (!forcedAfterFlight) {
				const forced: Promise<void> = inFlight
					.then(() => refresh(true))
					.finally(() => {
						if (forcedAfterFlight === forced) forcedAfterFlight = null;
					});
				forcedAfterFlight = forced;
			}
			return forcedAfterFlight;
		}
		const flight: Promise<void> = fetchCollections().finally(() => {
			if (inFlight === flight) inFlight = null;
		});
		inFlight = flight;
		return flight;
	}

	async function mutate<R extends Result>(request: Promise<R>): Promise<R> {
		const response = await request;
		if (response.success) await refresh(true);
		return response;
	}

	const store = {
		subscribe,
		load: refresh,

		async setDirectOnly(value: boolean) {
			if (directOnly === value) return;
			directOnly = value;
			await refresh(true);
		},

		async create(name: string, parentId?: string | null) {
			const response = await adapter.create(name, parentId);
			if (response.success) {
				const created = response.data?.collection;
				if (created) {
					update((state) =>
						state.collections.some((c) => c.id === created.id)
							? state
							: { ...state, collections: [...state.collections, created] }
					);
				}
				await refresh(true);
			}
			return response;
		},

		async move(id: string, parentId: string | null) {
			try {
				const response = await adapter.move(id, parentId);
				if (response.success) {
					await refresh(true);
				} else {
					logger.error(`Move ${label} failed:`, response.error ?? response.message ?? 'unknown');
				}
				return response;
			} catch (error) {
				logger.error(`Move ${label} failed:`, getErrorMessage(error));
				throw error;
			}
		},

		rename: (id: string, name: string) => mutate(adapter.rename(id, name)),
		remove: (id: string) => mutate(adapter.remove(id)),
		addMembers: (id: string, memberIds: string[]) => mutate(adapter.addMembers(id, memberIds)),

		reset() {
			lifetime++;
			loaded = false;
			inFlight = null;
			forcedAfterFlight = null;
			set(initialState);
		}
	};

	const { removeMembers, bulkMove } = adapter;
	return Object.assign(
		store,
		removeMembers && {
			removeMembers: (id: string, memberIds: string[]) => mutate(removeMembers.call(adapter, id, memberIds))
		},
		bulkMove && {
			async bulkMove(ids: string[], parentId: string | null): Promise<BulkMoveOutcome> {
				const response = await bulkMove.call(adapter, ids, parentId);
				if (response.success) await refresh(true);
				return {
					success: response.success,
					error: response.error,
					message: response.message,
					moved: response.data?.moved ?? 0,
					failed: response.data?.failed ?? ids.length,
					errors: response.data?.errors ?? []
				};
			}
		}
	) as unknown as CollectionTreeStore<A>;
}
