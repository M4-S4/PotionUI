import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { TabDraft, TabDraftOpsRequest, TabDraftSummary } from '$lib/services/api/tabDrafts';
import type { VariablesMap } from '$lib/utils/variableDefs';
import { createTabDraftSync, type TabDraftApiPort, type TabsPort } from './tabDraftSync';
import { adoptLegacyVariables, createTabDraftOutbox, type OutboxStorage } from './tabDraftOutbox';
import { TAB_DRAFT_OUTBOX_KEY } from './tabDraftKeys';
import { diffVariables, flattenDef, overlayOps, setOp, sameForSync } from './tabDraftOps';

const text = (value: string) => ({ type: 'text' as const, value });

function httpError(status: number, detail: Record<string, unknown> = {}) {
	return { response: { status, data: { detail: { success: false, ...detail } } } };
}

const networkError = () => new Error('Network Error');

function memoryStorage(): OutboxStorage & { data: Map<string, string> } {
	const data = new Map<string, string>();
	return {
		data,
		get: (key) => data.get(key) ?? null,
		set: (key, value) => void data.set(key, value)
	};
}

function fakeTabs(initial: Array<{ id: string; variables?: VariablesMap }>) {
	let tabs = initial;
	const listeners = new Set<(state: { tabs: typeof initial }) => void>();
	const emit = () => listeners.forEach((run) => run({ tabs }));
	const port: TabsPort = {
		subscribe(run) {
			listeners.add(run);
			run({ tabs });
			return () => listeners.delete(run);
		},
		setVariables(tabId, variables) {
			tabs = tabs.map((tab) => (tab.id === tabId ? { ...tab, variables } : tab));
			emit();
		}
	};
	return {
		port,
		variables: (id: string) => tabs.find((tab) => tab.id === id)?.variables,
		edit(id: string, variables: VariablesMap) {
			tabs = tabs.map((tab) => (tab.id === id ? { ...tab, variables } : tab));
			emit();
		},
		open(id: string) {
			tabs = [...tabs, { id }];
			emit();
		},
		close(id: string) {
			tabs = tabs.filter((tab) => tab.id !== id);
			emit();
		}
	};
}

function draft(tabId: string, variables: VariablesMap, rev = 1): TabDraft {
	return { tab_id: tabId, doc: { variables }, rev, created_at: null, updated_at: null };
}

function fakeApi(overrides: Partial<TabDraftApiPort> = {}) {
	const server = new Map<string, { variables: VariablesMap; rev: number }>();
	const calls = {
		list: vi.fn(),
		get: vi.fn(),
		replace: vi.fn(),
		apply: vi.fn(),
		remove: vi.fn()
	};
	const api: TabDraftApiPort = {
		async list() {
			calls.list();
			const data: TabDraftSummary[] = [...server.entries()].map(([tab_id, row]) => ({
				tab_id,
				variable_count: Object.keys(row.variables).length,
				rev: row.rev,
				updated_at: null
			}));
			return { success: true, data };
		},
		async get(tabId) {
			calls.get(tabId);
			const row = server.get(tabId);
			if (!row) throw httpError(404, { error: 'tab_draft_not_found' });
			return { success: true, data: draft(tabId, row.variables, row.rev) };
		},
		async replace(tabId, doc) {
			calls.replace(tabId, doc);
			const row = { variables: doc.variables ?? {}, rev: (server.get(tabId)?.rev ?? 0) + 1 };
			server.set(tabId, row);
			return { success: true, data: draft(tabId, row.variables, row.rev) };
		},
		async apply(tabId: string, request: TabDraftOpsRequest) {
			calls.apply(tabId, request);
			const current = server.get(tabId) ?? { variables: {}, rev: 0 };
			const variables = overlayOps(current.variables, request.ops);
			const row = { variables, rev: current.rev + 1 };
			server.set(tabId, row);
			return { success: true, data: draft(tabId, variables, row.rev) };
		},
		async remove(tabId) {
			calls.remove(tabId);
			if (!server.delete(tabId)) throw httpError(404, { error: 'tab_draft_not_found' });
		},
		...overrides
	};
	return { api, calls, server };
}

function setup(options: {
	tabs?: Array<{ id: string; variables?: VariablesMap }>;
	api?: Partial<TabDraftApiPort>;
	storage?: ReturnType<typeof memoryStorage>;
	seedServer?: Record<string, VariablesMap>;
	online?: () => boolean;
} = {}) {
	const storage = options.storage ?? memoryStorage();
	const outbox = createTabDraftOutbox(storage);
	const tabs = fakeTabs(options.tabs ?? [{ id: 't1' }]);
	const fake = fakeApi(options.api);
	for (const [id, variables] of Object.entries(options.seedServer ?? {})) {
		fake.server.set(id, { variables, rev: 3 });
	}
	const notify = { error: vi.fn(), warning: vi.fn() };
	const sync = createTabDraftSync({
		tabs: tabs.port,
		api: fake.api,
		outbox,
		notify,
		isOnline: options.online
	});
	return { sync, tabs, outbox, storage, notify, ...fake };
}

async function settle(ms = 0) {
	await vi.advanceTimersByTimeAsync(ms);
}

let windowStub: EventTarget;

beforeEach(() => {
	vi.useFakeTimers();
	windowStub = new EventTarget();
	vi.stubGlobal('window', windowStub);
});
afterEach(() => {
	vi.useRealTimers();
	vi.unstubAllGlobals();
});

describe('tabDraftOps', () => {
	it('diffs added, changed and removed variables into flat ops', () => {
		const prev: VariablesMap = { a: text('1'), b: text('2'), c: text('3') };
		const next: VariablesMap = { a: text('1'), b: text('two'), d: text('4') };
		expect(diffVariables(prev, next)).toEqual([
			{ op: 'remove', name: 'c' },
			{ op: 'set', name: 'b', type: 'text', value: 'two' },
			{ op: 'set', name: 'd', type: 'text', value: '4' }
		]);
	});

	it('flattens choice and number defs to the server shape', () => {
		const choice = { type: 'choice' as const, options: ['x', 'y'], mode: 'pin' as const, pinnedIndex: 1, linked: true };
		expect(setOp('c', choice)).toEqual({
			op: 'set',
			name: 'c',
			type: 'choice',
			options: ['x', 'y'],
			mode: 'pin',
			pinned_index: 1
		});
		const number = { type: 'number' as const, mode: 'shuffle' as const, min: 0, max: 5, step: 1, value: 2 };
		expect(flattenDef(number)).toEqual({ type: 'number', mode: 'shuffle', min: 0, max: 5, step: 1, value: 2 });
		expect(sameForSync(choice, { ...choice, linked: false })).toBe(true);
	});
});

describe('tabDraftOutbox', () => {
	it('keeps the newest op per variable name and only acks ops that were actually sent', () => {
		const outbox = createTabDraftOutbox(memoryStorage());
		outbox.putOps('t1', [setOp('a', text('1'))]);
		const sent = outbox.pendingOps('t1');
		outbox.putOps('t1', [setOp('a', text('2'))]);
		outbox.ackOps('t1', sent);
		expect(outbox.pendingOps('t1')).toEqual([setOp('a', text('2'))]);
	});

	it('persists across instances sharing storage', () => {
		const storage = memoryStorage();
		createTabDraftOutbox(storage).putOps('t1', [setOp('a', text('1'))]);
		expect(createTabDraftOutbox(storage).pendingOps('t1')).toEqual([setOp('a', text('1'))]);
	});

	it('queueDelete discards pending ops and the legacy stash for that tab', () => {
		const outbox = createTabDraftOutbox(memoryStorage());
		outbox.putOps('t1', [setOp('a', text('1'))]);
		outbox.stashLegacy('t1', { a: text('1') });
		outbox.queueDelete('t1');
		expect(outbox.pendingOps('t1')).toEqual([]);
		expect(outbox.legacyEntries()).toEqual({});
		expect(outbox.pendingDeletes()).toEqual(['t1']);
	});
});

describe('tabDraftSync write path', () => {
	it('batches rapid edits into one request after the debounce, one op per variable', async () => {
		const t = setup();
		t.sync.start();
		await settle();

		t.tabs.edit('t1', { a: text('1') });
		await settle(200);
		t.tabs.edit('t1', { a: text('12'), b: text('x') });
		await settle(200);
		t.tabs.edit('t1', { a: text('123'), b: text('x') });
		await settle(499);
		expect(t.calls.apply).not.toHaveBeenCalled();

		await settle(1);
		expect(t.calls.apply).toHaveBeenCalledTimes(1);
		const [tabId, request] = t.calls.apply.mock.calls[0];
		expect(tabId).toBe('t1');
		expect(request.ops).toEqual([
			{ op: 'set', name: 'a', type: 'text', value: '123' },
			{ op: 'set', name: 'b', type: 'text', value: 'x' }
		]);
		expect(t.outbox.pendingOps('t1')).toEqual([]);
	});

	it('writes ops to the outbox storage before any request is sent', async () => {
		const t = setup();
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { a: text('1') });
		const stored = JSON.parse(t.storage.data.get(TAB_DRAFT_OUTBOX_KEY)!);
		expect(stored.ops.t1.a).toEqual({ op: 'set', name: 'a', type: 'text', value: '1' });
	});

	it('sends removals as remove ops', async () => {
		const t = setup({ tabs: [{ id: 't1', variables: { a: text('1') } }], seedServer: { t1: { a: text('1') } } });
		t.sync.start();
		await settle();
		t.tabs.edit('t1', {});
		await settle(500);
		expect(t.calls.apply.mock.calls[0][1].ops).toEqual([{ op: 'remove', name: 'a' }]);
	});

	it('survives a reload: queued ops are restored into the tab and flushed by the next session', async () => {
		const storage = memoryStorage();
		const down = setup({
			storage,
			api: {
				list: async () => {
					throw networkError();
				},
				apply: async () => {
					throw networkError();
				}
			}
		});
		down.sync.start();
		await settle();
		down.tabs.edit('t1', { a: text('offline edit') });
		await settle(500);
		down.sync.stop();

		const next = setup({ storage });
		next.sync.start();
		expect(next.tabs.variables('t1')).toEqual({ a: text('offline edit') });
		await settle(600);
		expect(next.calls.apply).toHaveBeenCalledTimes(1);
		expect(next.calls.apply.mock.calls[0][1].ops).toEqual([
			{ op: 'set', name: 'a', type: 'text', value: 'offline edit' }
		]);
		expect(next.outbox.pendingOps('t1')).toEqual([]);
		expect(next.server.get('t1')?.variables).toEqual({ a: text('offline edit') });
	});

	it('retries with exponential backoff while the backend is down', async () => {
		let failures = 2;
		const t = setup({
			api: {
				apply: async (tabId, request) => {
					if (failures > 0) {
						failures -= 1;
						throw networkError();
					}
					return { success: true, data: draft(tabId, overlayOps({}, request.ops), 1) };
				}
			}
		});
		const apply = vi.spyOn(t.api, 'apply');
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { a: text('1') });
		await settle(500);
		expect(apply).toHaveBeenCalledTimes(1);
		await settle(999);
		expect(apply).toHaveBeenCalledTimes(1);
		await settle(1);
		expect(apply).toHaveBeenCalledTimes(2);
		await settle(1999);
		expect(apply).toHaveBeenCalledTimes(2);
		await settle(1);
		expect(apply).toHaveBeenCalledTimes(3);
		expect(t.outbox.pendingOps('t1')).toEqual([]);
	});

	it('does not send while offline', async () => {
		let online = false;
		const t = setup({ online: () => online });
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { a: text('1') });
		await settle(5000);
		expect(t.calls.apply).not.toHaveBeenCalled();
		expect(t.outbox.pendingOps('t1')).toHaveLength(1);
		online = true;
		windowStub.dispatchEvent(new Event('online'));
		await settle();
		expect(t.calls.apply).toHaveBeenCalledTimes(1);
		expect(t.calls.apply.mock.calls[0][1].ops).toEqual([setOp('a', text('1'))]);
	});

	it('flushes queued ops immediately on the online event, without waiting for the debounce', async () => {
		let online = false;
		const t = setup({ online: () => online });
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { a: text('1') });
		await settle(100);
		online = true;
		windowStub.dispatchEvent(new Event('online'));
		await settle();
		expect(t.calls.apply).toHaveBeenCalledTimes(1);
	});

	it('splits more than 500 variables into requests of 500 and 1', async () => {
		const t = setup();
		t.sync.start();
		await settle();
		const many: VariablesMap = {};
		for (let i = 0; i < 501; i++) many[`v${i}`] = text(String(i));
		t.tabs.edit('t1', many);
		await settle(500);
		expect(t.calls.apply).toHaveBeenCalledTimes(2);
		expect(t.calls.apply.mock.calls.map((call) => call[1].ops.length)).toEqual([500, 1]);
		expect(t.outbox.pendingOps('t1')).toEqual([]);
	});

	it.each([500, 401, 403, 404, 408, 429])('keeps ops queued and retries after a %i', async (status) => {
		let failing = true;
		const t = setup({
			api: {
				apply: async (tabId, request) => {
					if (failing) throw httpError(status);
					return { success: true, data: draft(tabId, overlayOps({}, request.ops), 1) };
				}
			}
		});
		const apply = vi.spyOn(t.api, 'apply');
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { a: text('1') });
		await settle(500);
		expect(apply).toHaveBeenCalledTimes(1);
		expect(t.outbox.pendingOps('t1')).toHaveLength(1);
		expect(t.notify.error).not.toHaveBeenCalled();

		failing = false;
		await settle(1000);
		expect(apply).toHaveBeenCalledTimes(2);
		expect(t.outbox.pendingOps('t1')).toEqual([]);
	});

	it('drops a non-transient 4xx other than 413 and 422 with an error toast and no retry', async () => {
		const t = setup({
			api: {
				apply: async () => {
					throw httpError(400, { message: 'Bad request' });
				}
			}
		});
		const apply = vi.spyOn(t.api, 'apply');
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { a: text('1') });
		await settle(60000);
		expect(apply).toHaveBeenCalledTimes(1);
		expect(t.notify.error).toHaveBeenCalledTimes(1);
		expect(t.outbox.pendingOps('t1')).toEqual([]);
		expect(t.tabs.variables('t1')).toEqual({ a: text('1') });
	});
});

describe('tabDraftSync read path and merge', () => {
	it('hydrates a tab from its server draft without writing anything back', async () => {
		const t = setup({ seedServer: { t1: { style: text('oil') } } });
		t.sync.start();
		await settle();
		expect(t.tabs.variables('t1')).toEqual({ style: text('oil') });
		await settle(2000);
		expect(t.calls.apply).not.toHaveBeenCalled();
		expect(t.calls.replace).not.toHaveBeenCalled();
		expect(t.sync.revOf('t1')).toBe(3);
	});

	it('treats a missing draft as a tab with no variables and creates nothing', async () => {
		const t = setup({
			api: {
				list: async () => ({ success: true, data: [{ tab_id: 't1', variable_count: 0, rev: 1, updated_at: null }] })
			}
		});
		t.sync.start();
		await settle(2000);
		expect(t.calls.get).toHaveBeenCalledWith('t1');
		expect(t.tabs.variables('t1') ?? {}).toEqual({});
		expect(t.calls.apply).not.toHaveBeenCalled();
		expect(t.calls.replace).not.toHaveBeenCalled();
	});

	it('does not fetch or create anything for a tab the server has never seen', async () => {
		const t = setup();
		t.sync.start();
		await settle(2000);
		expect(t.calls.get).not.toHaveBeenCalled();
		expect(t.calls.apply).not.toHaveBeenCalled();
		expect(t.calls.replace).not.toHaveBeenCalled();
	});

	it('keeps a newer local edit made while a request was in flight', async () => {
		let release!: () => void;
		const gate = new Promise<void>((resolve) => (release = resolve));
		let first = true;
		const t = setup({
			api: {
				apply: async (tabId, request) => {
					if (first) {
						first = false;
						await gate;
						return { success: true, data: draft(tabId, { a: text('old'), other: text('theirs') }, 7) };
					}
					return { success: true, data: draft(tabId, overlayOps({ other: text('theirs') }, request.ops), 8) };
				}
			}
		});
		const apply = vi.spyOn(t.api, 'apply');
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { a: text('old') });
		await settle(500);
		expect(apply).toHaveBeenCalledTimes(1);

		t.tabs.edit('t1', { a: text('newer') });
		release();
		await settle();

		expect(t.tabs.variables('t1')).toEqual({ a: text('newer'), other: text('theirs') });
		expect(apply).toHaveBeenCalledTimes(2);
		expect(apply.mock.calls[1][1].ops).toEqual([setOp('a', text('newer'))]);
		expect(t.sync.revOf('t1')).toBe(8);
	});

	it('keeps the local def object when the server copy differs only in fields it does not store', async () => {
		const linked = { type: 'choice' as const, options: ['x', 'y'], mode: 'shuffle' as const, pinnedIndex: null, linked: true };
		const t = setup();
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { c: linked });
		await settle(500);
		expect(t.tabs.variables('t1')!.c).toBe(linked);
	});
});

describe('tabDraftSync tab close', () => {
	it('sends DELETE when a tab is closed and drops its pending ops', async () => {
		const t = setup({
			tabs: [{ id: 't1' }, { id: 't2' }],
			seedServer: { t1: { a: text('1') } }
		});
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { a: text('2') });
		t.tabs.close('t1');
		await settle(500);
		expect(t.calls.remove).toHaveBeenCalledWith('t1');
		expect(t.calls.apply).not.toHaveBeenCalled();
		expect(t.outbox.pendingDeletes()).toEqual([]);
	});

	it('queues the delete in the outbox while offline and sends it later', async () => {
		let failing = true;
		const t = setup({
			tabs: [{ id: 't1' }, { id: 't2' }],
			seedServer: { t1: { a: text('1') } },
			api: {
				remove: async () => {
					if (failing) throw networkError();
				}
			}
		});
		const remove = vi.spyOn(t.api, 'remove');
		t.sync.start();
		await settle();
		t.tabs.close('t1');
		await settle(500);
		expect(remove).toHaveBeenCalledTimes(1);
		expect(t.outbox.pendingDeletes()).toEqual(['t1']);

		failing = false;
		await settle(1000);
		expect(remove).toHaveBeenCalledTimes(2);
		expect(t.outbox.pendingDeletes()).toEqual([]);
	});

	it('treats a 404 on delete as done but keeps the delete queued after a 500', async () => {
		const t = setup({
			tabs: [{ id: 't1' }, { id: 't2' }, { id: 't3' }],
			api: {
				remove: async (tabId) => {
					throw tabId === 't1' ? httpError(404) : httpError(500);
				}
			}
		});
		t.sync.start();
		await settle();
		t.tabs.close('t1');
		await settle(500);
		expect(t.outbox.pendingDeletes()).toEqual([]);

		t.tabs.close('t2');
		await settle(500);
		expect(t.outbox.pendingDeletes()).toEqual(['t2']);
	});
});

describe('tabDraftSync migration', () => {
	const legacy: VariablesMap = { style: text('oil'), mood: text('calm') };

	it('uploads local variables with PUT once, then never again', async () => {
		const storage = memoryStorage();
		const first = setup({ storage });
		adoptLegacyVariables([{ id: 't1', variables: legacy }], first.outbox);
		const seeded = setup({ storage, tabs: [{ id: 't1', variables: legacy }] });
		seeded.sync.start();
		await settle();
		expect(seeded.calls.replace).toHaveBeenCalledTimes(1);
		expect(seeded.calls.replace).toHaveBeenCalledWith('t1', { variables: legacy });
		expect(seeded.calls.apply).not.toHaveBeenCalled();
		expect(seeded.outbox.isMigrated()).toBe(true);
		expect(seeded.outbox.legacyEntries()).toEqual({});
		seeded.sync.stop();

		const again = setup({ storage, tabs: [{ id: 't1', variables: legacy }], seedServer: { t1: legacy } });
		adoptLegacyVariables([{ id: 't1', variables: { stale: text('x') } }], again.outbox);
		expect(again.outbox.legacyEntries()).toEqual({});
		again.sync.start();
		await settle(1000);
		expect(again.calls.replace).not.toHaveBeenCalled();
	});

	it('prefers an existing server draft over the local copy', async () => {
		const t = setup({ tabs: [{ id: 't1', variables: legacy }], seedServer: { t1: { fresh: text('1') } } });
		adoptLegacyVariables([{ id: 't1', variables: legacy }], t.outbox);
		t.sync.start();
		await settle();
		expect(t.calls.replace).not.toHaveBeenCalled();
		expect(t.tabs.variables('t1')).toEqual({ fresh: text('1') });
	});

	it('keeps the variables and warns when the draft limit is reached, then retries them as ops', async () => {
		let limited = true;
		const t = setup({
			tabs: [{ id: 't1', variables: legacy }],
			api: {
				replace: async () => {
					throw httpError(409, { error: 'tab_draft_limit', limit: 200 });
				},
				apply: async (tabId, request) => {
					if (limited) throw httpError(409, { error: 'tab_draft_limit', limit: 200 });
					return { success: true, data: draft(tabId, overlayOps({}, request.ops), 1) };
				}
			}
		});
		adoptLegacyVariables([{ id: 't1', variables: legacy }], t.outbox);
		t.sync.start();
		await settle();

		expect(t.notify.warning).toHaveBeenCalledTimes(1);
		expect(t.tabs.variables('t1')).toEqual(legacy);
		expect(t.outbox.pendingOps('t1').map((op) => op.name).sort()).toEqual(['mood', 'style']);
		expect(t.outbox.legacyEntries()).toEqual({});
		expect(t.outbox.isMigrated()).toBe(true);

		limited = false;
		await settle(30000);
		expect(t.outbox.pendingOps('t1')).toEqual([]);
	});

	it('keeps the stash and retries when the backend is unreachable', async () => {
		let down = true;
		const t = setup({
			tabs: [{ id: 't1', variables: legacy }],
			api: {
				list: async () => {
					if (down) throw networkError();
					return { success: true, data: [] };
				}
			}
		});
		adoptLegacyVariables([{ id: 't1', variables: legacy }], t.outbox);
		t.sync.start();
		await settle();
		expect(t.outbox.isMigrated()).toBe(false);
		expect(t.outbox.legacyEntries()).toEqual({ t1: legacy });

		down = false;
		await settle(1000);
		expect(t.calls.replace).toHaveBeenCalledTimes(1);
		expect(t.outbox.isMigrated()).toBe(true);
	});
});

describe('tabDraftSync error handling', () => {
	it('toasts a 413, keeps the edit visible and does not retry', async () => {
		const t = setup({
			api: {
				apply: async () => {
					throw httpError(413, { error: 'tab_draft_too_large', limit: 262144 });
				}
			}
		});
		const apply = vi.spyOn(t.api, 'apply');
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { big: text('x') });
		await settle(500);
		expect(apply).toHaveBeenCalledTimes(1);
		await settle(60000);
		expect(apply).toHaveBeenCalledTimes(1);
		expect(t.notify.error).toHaveBeenCalledTimes(1);
		expect(t.notify.error.mock.calls[0][0]).toContain('"big"');
		expect(t.notify.error.mock.calls[0][0]).toContain('too large');
		expect(t.tabs.variables('t1')).toEqual({ big: text('x') });
		expect(t.outbox.pendingOps('t1')).toEqual([]);
	});

	it('rejects only the invalid ops of a 422 batch, names the problem and still syncs the rest', async () => {
		const t = setup({
			api: {
				apply: async (tabId, request) => {
					const badIndex = request.ops.findIndex((op) => op.name === 'bad');
					if (badIndex >= 0) {
						throw httpError(422, {
							error: 'tab_draft_invalid',
							message: 'Some operations are invalid.',
							operations: [{ index: badIndex, name: 'bad', op: 'set', errors: ["'bad': invalid type."] }]
						});
					}
					return { success: true, data: draft(tabId, overlayOps({}, request.ops), 1) };
				}
			}
		});
		const apply = vi.spyOn(t.api, 'apply');
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { bad: text('1'), good: text('2') });
		await settle(500);

		expect(t.notify.error).toHaveBeenCalledTimes(1);
		expect(t.notify.error.mock.calls[0][0]).toContain('"bad"');
		expect(t.notify.error.mock.calls[0][0]).toContain('invalid type');
		expect(apply).toHaveBeenCalledTimes(2);
		expect(apply.mock.calls[1][1].ops).toEqual([setOp('good', text('2'))]);
		expect(t.tabs.variables('t1')).toEqual({ bad: text('1'), good: text('2') });
		expect(t.outbox.pendingOps('t1')).toEqual([]);
	});

	it('sends a rejected variable again once it is edited', async () => {
		let reject = true;
		const t = setup({
			api: {
				apply: async (tabId, request) => {
					if (reject) throw httpError(422, { operations: [{ index: 0, name: 'a', errors: ['nope'] }] });
					return { success: true, data: draft(tabId, overlayOps({}, request.ops), 1) };
				}
			}
		});
		const apply = vi.spyOn(t.api, 'apply');
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { a: text('1') });
		await settle(500);

		reject = false;
		t.tabs.edit('t1', { a: text('2') });
		await settle(500);
		expect(apply.mock.calls.at(-1)![1].ops).toEqual([setOp('a', text('2'))]);
		expect(t.outbox.pendingOps('t1')).toEqual([]);
	});

	it('warns once on a 409, keeps the ops queued and retries later', async () => {
		let limited = true;
		const t = setup({
			api: {
				apply: async (tabId, request) => {
					if (limited) throw httpError(409, { error: 'tab_draft_limit', limit: 200 });
					return { success: true, data: draft(tabId, overlayOps({}, request.ops), 1) };
				}
			}
		});
		const apply = vi.spyOn(t.api, 'apply');
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { a: text('1') });
		await settle(500);
		await settle(1000);
		expect(apply.mock.calls.length).toBeGreaterThanOrEqual(2);
		expect(t.notify.warning).toHaveBeenCalledTimes(1);
		expect(t.notify.warning.mock.calls[0][0]).toContain('Close old tabs');
		expect(t.outbox.pendingOps('t1')).toHaveLength(1);

		limited = false;
		await settle(30000);
		expect(t.outbox.pendingOps('t1')).toEqual([]);
	});
});

describe('tabDraftSync boot ordering', () => {
	it('holds edits made during a slow boot until the migration upload has finished', async () => {
		let release!: () => void;
		const gate = new Promise<void>((resolve) => (release = resolve));
		const legacy: VariablesMap = { style: text('old') };
		const order: string[] = [];
		const t = setup({
			tabs: [{ id: 't1', variables: legacy }],
			api: {
				list: async () => {
					await gate;
					return { success: true, data: [] };
				}
			}
		});
		adoptLegacyVariables([{ id: 't1', variables: legacy }], t.outbox);
		const replace = t.api.replace.bind(t.api);
		t.api.replace = async (tabId, doc) => {
			order.push('replace');
			return replace(tabId, doc);
		};
		const apply = t.api.apply.bind(t.api);
		t.api.apply = async (tabId, request) => {
			order.push('apply');
			return apply(tabId, request);
		};
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { style: text('newer') });
		await settle(5000);
		expect(order).toEqual([]);

		release();
		await settle(1000);
		expect(order).toEqual(['replace', 'apply']);
		expect(t.server.get('t1')?.variables).toEqual({ style: text('newer') });
	});

	it('does not queue deletes when the tab list starts empty and then fills, or across stop and start', async () => {
		const t = setup({ tabs: [{ id: 't1' }], seedServer: { t1: { a: text('1') } } });
		t.sync.start();
		await settle(1000);
		t.sync.stop();
		t.sync.start();
		await settle(1000);
		expect(t.outbox.pendingDeletes()).toEqual([]);
		expect(t.calls.remove).not.toHaveBeenCalled();

		const listeners = fakeTabs([]);
		const outbox = createTabDraftOutbox(memoryStorage());
		const fake = fakeApi();
		const sync = createTabDraftSync({
			tabs: listeners.port,
			api: fake.api,
			outbox,
			notify: { error: vi.fn(), warning: vi.fn() }
		});
		sync.start();
		await settle(1000);
		listeners.open('t1');
		await settle(1000);
		expect(outbox.pendingDeletes()).toEqual([]);
		expect(fake.calls.remove).not.toHaveBeenCalled();
	});

	it('does not delete a tab that the first emission already contained', async () => {
		const t = setup({ tabs: [{ id: 't1' }, { id: 't2' }] });
		t.sync.start();
		await settle(2000);
		expect(t.calls.remove).not.toHaveBeenCalled();
	});
});

describe('tabDraftSync boot retry', () => {
	it('retries a failing list with backoff and on the online event, keeping edits queued', async () => {
		let failures = 2;
		const t = setup({
			api: {
				list: async () => {
					if (failures > 0) {
						failures -= 1;
						throw networkError();
					}
					return { success: true, data: [] };
				}
			}
		});
		const list = vi.spyOn(t.api, 'list');
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { a: text('1') });
		await settle(999);
		expect(list).toHaveBeenCalledTimes(1);
		await settle(1);
		expect(list).toHaveBeenCalledTimes(2);
		expect(t.calls.apply).not.toHaveBeenCalled();
		expect(t.outbox.pendingOps('t1')).toHaveLength(1);

		windowStub.dispatchEvent(new Event('online'));
		await settle();
		expect(list).toHaveBeenCalledTimes(3);
		await settle(500);
		expect(t.calls.apply).toHaveBeenCalledTimes(1);
		expect(t.outbox.pendingOps('t1')).toEqual([]);
	});

	it('caps the boot retry delay at the maximum backoff', async () => {
		const t = setup({
			api: {
				list: async () => {
					throw networkError();
				}
			}
		});
		const list = vi.spyOn(t.api, 'list');
		t.sync.start();
		await settle(1000 + 2000 + 4000 + 8000 + 16000);
		expect(list).toHaveBeenCalledTimes(6);
		await settle(29999);
		expect(list).toHaveBeenCalledTimes(6);
		await settle(1);
		expect(list).toHaveBeenCalledTimes(7);
	});
});

describe('tabDraftSync storage failures', () => {
	function failingStorage(): OutboxStorage {
		return {
			get: () => null,
			set: () => {
				throw new Error('QuotaExceededError');
			}
		};
	}

	it('reports a failed legacy stash and retains the variables for persistence', () => {
		const outbox = createTabDraftOutbox(failingStorage());
		const legacy = { style: text('old') };
		adoptLegacyVariables([{ id: 't1', variables: legacy }], outbox);
		expect(outbox.retainedLegacyIds().has('t1')).toBe(true);
		expect(outbox.legacyEntries()).toEqual({ t1: legacy });
		outbox.clearLegacy('t1');
		expect(outbox.retainedLegacyIds().has('t1')).toBe(false);
	});

	it('warns once when outbox writes fail but still syncs the edit', async () => {
		const outbox = createTabDraftOutbox(failingStorage());
		const tabs = fakeTabs([{ id: 't1' }]);
		const fake = fakeApi();
		const notify = { error: vi.fn(), warning: vi.fn() };
		const sync = createTabDraftSync({ tabs: tabs.port, api: fake.api, outbox, notify });
		sync.start();
		await settle();
		tabs.edit('t1', { a: text('1') });
		tabs.edit('t1', { a: text('2') });
		await settle(500);
		expect(notify.warning).toHaveBeenCalledTimes(1);
		expect(fake.calls.apply).toHaveBeenCalledTimes(1);
		expect(fake.server.get('t1')?.variables).toEqual({ a: text('2') });
	});
});

describe('tabDraftSync stop', () => {
	it('sends nothing for the old session when stopped during a flush', async () => {
		let release!: () => void;
		const gate = new Promise<void>((resolve) => (release = resolve));
		const t = setup({
			tabs: [{ id: 't1' }, { id: 't2' }],
			api: {
				apply: async (tabId, request) => {
					await gate;
					return { success: true, data: draft(tabId, overlayOps({}, request.ops), 1) };
				}
			}
		});
		const apply = vi.spyOn(t.api, 'apply');
		t.sync.start();
		await settle();
		t.tabs.edit('t1', { a: text('1') });
		t.tabs.edit('t2', { b: text('2') });
		await settle(500);
		expect(apply).toHaveBeenCalledTimes(1);

		t.sync.stop();
		release();
		await settle(60000);
		expect(apply).toHaveBeenCalledTimes(1);
		expect(t.tabs.variables('t1')).toEqual({ a: text('1') });
	});

	it('stops reacting to edits after stop()', async () => {
		const t = setup();
		t.sync.start();
		await settle();
		t.sync.stop();
		t.tabs.edit('t1', { a: text('1') });
		await settle(2000);
		expect(t.calls.apply).not.toHaveBeenCalled();
	});
});
