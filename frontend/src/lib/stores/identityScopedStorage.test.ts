import { describe, it, expect, vi, afterEach } from 'vitest';
import { TABS_STORAGE_KEY } from '$lib/types/tabs';
import {
	isDifferentIdentity,
	keysToPurge,
	namespacedKeys,
	identityKey,
	hasIdentityState,
	removeIdentityKeys,
	scopedStorage,
	LAST_USER_ID_KEY
} from './identityScopedStorage';
import { TAB_DRAFT_OUTBOX_KEY, TAB_DRAFT_MIGRATED_KEY } from './tabDraftKeys';
import { createTabDraftOutbox } from './tabDraftOutbox';
import { setOp } from './tabDraftOps';

describe('isDifferentIdentity', () => {
	it('is false when no prior identity is recorded (first login)', () => {
		expect(isDifferentIdentity(null, 'user-a')).toBe(false);
	});

	it('is false when the same user logs back in', () => {
		expect(isDifferentIdentity('user-a', 'user-a')).toBe(false);
	});

	it('is true when a different user logs in', () => {
		expect(isDifferentIdentity('user-a', 'user-b')).toBe(true);
	});
});

describe('keysToPurge', () => {
	it('purges the tab state key', () => {
		expect(keysToPurge([TABS_STORAGE_KEY])).toEqual([TABS_STORAGE_KEY]);
	});

	it('purges every unified-ai-chat identity key', () => {
		const keys = [
			'unified-ai-chat-session-id',
			'unified-ai-chat-config-id',
			'unified-ai-chat-pinned-tab',
			'unified-ai-chat-attach-image',
			'unified-ai-chat-enable-tools',
			'phrasebook-generation-config'
		];
		expect(keysToPurge(keys)).toEqual(keys);
	});

	it('purges per-mode disabled-tools keys by prefix', () => {
		expect(keysToPurge(['unified-ai-chat-disabled-tools:chat', 'unified-ai-chat-disabled-tools:agent'])).toEqual([
			'unified-ai-chat-disabled-tools:chat',
			'unified-ai-chat-disabled-tools:agent'
		]);
	});

	it('leaves device/UI preference keys untouched', () => {
		const uiKeys = [
			'auth_token',
			'remember_me',
			'theme',
			'unified-ai-chat-history-rail-collapsed',
			'generation-panel-active-drawer',
			'potionui-form-audience',
			'audio_player_volume',
			'autoSaveEnabled',
			'autoSaveInterval'
		];
		expect(keysToPurge(uiKeys)).toEqual([]);
	});

	it('purges only the matching subset out of a mixed key set', () => {
		const mixed = [TABS_STORAGE_KEY, 'theme', 'unified-ai-chat-config-id', 'audio_player_volume'];
		expect(keysToPurge(mixed)).toEqual([TABS_STORAGE_KEY, 'unified-ai-chat-config-id']);
	});
});

describe('tab draft keys are identity scoped', () => {
	afterEach(() => vi.unstubAllGlobals());

	function stubLocalStorage(initial: Record<string, string>) {
		const data = new Map(Object.entries(initial));
		vi.stubGlobal('localStorage', {
			get length() {
				return data.size;
			},
			key: (index: number) => [...data.keys()][index] ?? null,
			getItem: (key: string) => data.get(key) ?? null,
			setItem: (key: string, value: string) => void data.set(key, value),
			removeItem: (key: string) => void data.delete(key)
		});
		return data;
	}

	it('purges and namespaces the outbox and migration flag', () => {
		expect(keysToPurge([TAB_DRAFT_OUTBOX_KEY, TAB_DRAFT_MIGRATED_KEY])).toEqual([
			TAB_DRAFT_OUTBOX_KEY,
			TAB_DRAFT_MIGRATED_KEY
		]);
		const scoped = [identityKey(TAB_DRAFT_OUTBOX_KEY, 'a'), identityKey(TAB_DRAFT_MIGRATED_KEY, 'b')];
		expect(namespacedKeys(scoped)).toEqual(scoped);
		expect(namespacedKeys(scoped, 'a')).toEqual([scoped[0]]);
	});

	it('keeps one user outbox out of another user scope and removes it with the account', () => {
		const data = stubLocalStorage({ [LAST_USER_ID_KEY]: 'user-a' });
		createTabDraftOutbox(scopedStorage).putOps('t1', [setOp('style', { type: 'text', value: 'x' })]);
		expect(data.has(identityKey(TAB_DRAFT_OUTBOX_KEY, 'user-a'))).toBe(true);

		data.set(LAST_USER_ID_KEY, 'user-b');
		expect(createTabDraftOutbox(scopedStorage).pendingOps('t1')).toEqual([]);

		expect(hasIdentityState('user-a')).toBe(true);
		removeIdentityKeys('user-a');
		expect(data.has(identityKey(TAB_DRAFT_OUTBOX_KEY, 'user-a'))).toBe(false);
	});
});
