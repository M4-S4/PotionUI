import type { TabDraftOp } from '$lib/services/api/tabDrafts';
import type { VariablesMap } from '$lib/utils/variableDefs';
import { scopedStorage } from '$lib/stores/identityScopedStorage';
import { deepEqual } from './tabDraftOps';
import { TAB_DRAFT_OUTBOX_KEY, TAB_DRAFT_MIGRATED_KEY } from './tabDraftKeys';


export interface OutboxState {
	ops: Record<string, Record<string, TabDraftOp>>;
	deletes: string[];
	legacy: Record<string, VariablesMap>;
}

export interface OutboxStorage {
	get(key: string): string | null;
	set(key: string, value: string): void;
}

function emptyState(): OutboxState {
	return { ops: {}, deletes: [], legacy: {} };
}

function parseState(raw: string | null): OutboxState {
	if (!raw) return emptyState();
	try {
		const parsed = JSON.parse(raw) as Partial<OutboxState>;
		return {
			ops: parsed.ops && typeof parsed.ops === 'object' ? parsed.ops : {},
			deletes: Array.isArray(parsed.deletes) ? parsed.deletes : [],
			legacy: parsed.legacy && typeof parsed.legacy === 'object' ? parsed.legacy : {}
		};
	} catch {
		return emptyState();
	}
}

export function createTabDraftOutbox(storage: OutboxStorage) {
	let cache: OutboxState | null = null;

	function state(): OutboxState {
		if (!cache) {
			try {
				cache = parseState(storage.get(TAB_DRAFT_OUTBOX_KEY));
			} catch {
				cache = emptyState();
			}
		}
		return cache;
	}

	const retained = new Set<string>();

	function persist(): boolean {
		try {
			storage.set(TAB_DRAFT_OUTBOX_KEY, JSON.stringify(state()));
			return true;
		} catch {
			return false;
		}
	}

	return {
		reset(): void {
			cache = null;
		},

		isMigrated(): boolean {
			try {
				return storage.get(TAB_DRAFT_MIGRATED_KEY) === '1';
			} catch {
				return false;
			}
		},

		markMigrated(): void {
			try {
				storage.set(TAB_DRAFT_MIGRATED_KEY, '1');
			} catch {
				return;
			}
		},

		putOps(tabId: string, ops: TabDraftOp[]): boolean {
			if (ops.length === 0) return true;
			const current = state();
			const bucket = (current.ops[tabId] ??= {});
			for (const op of ops) bucket[op.name] = op;
			return persist();
		},

		pendingOps(tabId: string): TabDraftOp[] {
			return Object.values(state().ops[tabId] ?? {});
		},

		pendingTabIds(): string[] {
			return Object.keys(state().ops).filter((id) => Object.keys(state().ops[id]).length > 0);
		},

		ackOps(tabId: string, sent: TabDraftOp[]): void {
			const bucket = state().ops[tabId];
			if (!bucket) return;
			for (const op of sent) {
				if (deepEqual(bucket[op.name], op)) delete bucket[op.name];
			}
			if (Object.keys(bucket).length === 0) delete state().ops[tabId];
			persist();
		},

		queueDelete(tabId: string): void {
			const current = state();
			delete current.ops[tabId];
			delete current.legacy[tabId];
			retained.delete(tabId);
			if (!current.deletes.includes(tabId)) current.deletes.push(tabId);
			persist();
		},

		pendingDeletes(): string[] {
			return [...state().deletes];
		},

		ackDelete(tabId: string): void {
			const current = state();
			current.deletes = current.deletes.filter((id) => id !== tabId);
			persist();
		},

		stashLegacy(tabId: string, variables: VariablesMap): boolean {
			const current = state();
			if (current.legacy[tabId]) return true;
			current.legacy[tabId] = variables;
			const saved = persist();
			if (!saved) retained.add(tabId);
			return saved;
		},

		retainedLegacyIds(): ReadonlySet<string> {
			return retained;
		},

		legacyEntries(): Record<string, VariablesMap> {
			return { ...state().legacy };
		},

		clearLegacy(tabId: string): void {
			const current = state();
			retained.delete(tabId);
			if (!(tabId in current.legacy)) return;
			delete current.legacy[tabId];
			persist();
		}
	};
}

export type TabDraftOutbox = ReturnType<typeof createTabDraftOutbox>;

export const tabDraftOutbox = createTabDraftOutbox(scopedStorage);

export function adoptLegacyVariables(
	persistedTabs: ReadonlyArray<{ id: string; variables?: VariablesMap }>,
	outbox: TabDraftOutbox = tabDraftOutbox
): Record<string, VariablesMap> {
	if (outbox.isMigrated()) return {};
	for (const tab of persistedTabs) {
		const variables = tab.variables;
		if (!variables || typeof variables !== 'object' || Object.keys(variables).length === 0) continue;
		outbox.stashLegacy(tab.id, variables);
	}
	return outbox.legacyEntries();
}
