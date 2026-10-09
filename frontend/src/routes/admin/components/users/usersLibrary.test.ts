import { describe, it, expect } from 'vitest';
import {
	USERS_LIBRARY_SECTIONS,
	accountTypeCounts,
	deletableGroupIds,
	deletableUserIds,
	groupEditFormFrom,
	groupUpdatePayload,
	pluralize,
	userEditFormFrom,
	userUpdatePayload,
	usersSubViewFromParam
} from './usersLibrary';
import type { User } from '$lib/stores/auth';
import type { UserGroup } from '$lib/services/admin-api';

function makeUser(overrides: Partial<User>): User {
	return {
		id: 'u1',
		username: 'user',
		email: 'user@example.com',
		account_type: 'USER',
		created_at: null,
		last_login: null,
		avatar_url: null,
		has_local_password: true,
		...overrides
	};
}

function makeGroup(overrides: Partial<UserGroup>): UserGroup {
	return {
		id: 'g1',
		name: 'group',
		is_system: false,
		...overrides
	};
}

describe('accountTypeCounts', () => {
	it('counts admins and regular users separately', () => {
		const users = [
			makeUser({ id: '1', account_type: 'ADMIN' }),
			makeUser({ id: '2', account_type: 'USER' }),
			makeUser({ id: '3', account_type: 'USER' })
		];
		expect(accountTypeCounts(users)).toEqual({ ADMIN: 1, USER: 2 });
	});

	it('returns zero counts for an empty list', () => {
		expect(accountTypeCounts([])).toEqual({ ADMIN: 0, USER: 0 });
	});
});

describe('deletableGroupIds', () => {
	it('excludes built-in groups from the selected set', () => {
		const groups = [makeGroup({ id: 'a', is_system: true }), makeGroup({ id: 'b', is_system: false }), makeGroup({ id: 'c', is_system: false })];
		const result = deletableGroupIds(groups, new Set(['a', 'b', 'c']));
		expect(result.deletableIds.sort()).toEqual(['b', 'c']);
		expect(result.skippedCount).toBe(1);
	});

	it('ignores groups outside the selection', () => {
		const groups = [makeGroup({ id: 'a', is_system: false }), makeGroup({ id: 'b', is_system: false })];
		const result = deletableGroupIds(groups, new Set(['a']));
		expect(result.deletableIds).toEqual(['a']);
		expect(result.skippedCount).toBe(0);
	});
});

describe('deletableUserIds', () => {
	it('excludes the current user from a bulk delete', () => {
		const result = deletableUserIds(new Set(['1', '2', '3']), '2');
		expect(result.deletableIds.sort()).toEqual(['1', '3']);
		expect(result.skippedCount).toBe(1);
	});

	it('keeps every id when the current user is not selected', () => {
		const result = deletableUserIds(new Set(['1', '3']), '2');
		expect(result.deletableIds.sort()).toEqual(['1', '3']);
		expect(result.skippedCount).toBe(0);
	});
});

describe('users sub views', () => {
	it('lists Plans as the third section', () => {
		expect(USERS_LIBRARY_SECTIONS.map((s) => s.id)).toEqual(['users', 'groups', 'plans']);
	});

	it('reads the view param and falls back to users', () => {
		expect(usersSubViewFromParam('plans')).toBe('plans');
		expect(usersSubViewFromParam('groups')).toBe('groups');
		expect(usersSubViewFromParam('nope')).toBe('users');
		expect(usersSubViewFromParam(null)).toBe('users');
	});
});

describe('userEditFormFrom', () => {
	it('copies identity fields and blanks the password', () => {
		expect(userEditFormFrom(makeUser({ username: 'a', email: 'a@x.io', account_type: 'ADMIN' }))).toEqual({
			username: 'a',
			email: 'a@x.io',
			password: '',
			account_type: 'ADMIN'
		});
	});

	it('returns an empty regular-user form for null', () => {
		expect(userEditFormFrom(null)).toEqual({ username: '', email: '', password: '', account_type: 'USER' });
	});
});

describe('userUpdatePayload', () => {
	it('omits empty fields so a blank password keeps the current one', () => {
		expect(userUpdatePayload({ username: 'a', email: 'a@x.io', password: '', account_type: 'USER' })).toEqual({
			username: 'a',
			email: 'a@x.io',
			account_type: 'USER'
		});
	});

	it('includes a typed password', () => {
		expect(userUpdatePayload({ username: 'a', email: 'a@x.io', password: 'pw', account_type: 'USER' }).password).toBe('pw');
	});
});

describe('groupEditFormFrom', () => {
	it('keeps a valid content policy and defaults the description', () => {
		expect(groupEditFormFrom({ name: 'g', description: null, content_policy: 'blur' })).toEqual({
			name: 'g',
			description: '',
			content_policy: 'blur'
		});
	});

	it('falls back to inherit for unknown policies and for null', () => {
		expect(groupEditFormFrom({ name: 'g', content_policy: 'weird' }).content_policy).toBe('inherit');
		expect(groupEditFormFrom(null)).toEqual({ name: '', description: '', content_policy: 'inherit' });
	});
});

describe('groupUpdatePayload', () => {
	it('sends the policy, mapping inherit to null', () => {
		expect(groupUpdatePayload({ name: 'g', description: 'd', content_policy: 'inherit' }, false)).toEqual({
			name: 'g',
			description: 'd',
			content_policy: null
		});
		expect(groupUpdatePayload({ name: 'g', description: 'd', content_policy: 'blocked' }, false).content_policy).toBe('blocked');
	});

	it('leaves the policy out for locked groups and omits an empty name', () => {
		expect(groupUpdatePayload({ name: '', description: 'd', content_policy: 'blocked' }, true)).toEqual({ description: 'd' });
	});
});

describe('pluralize', () => {
	it('uses the singular only for one', () => {
		expect(pluralize(1, 'user')).toBe('1 user');
		expect(pluralize(0, 'user')).toBe('0 users');
		expect(pluralize(2, 'change')).toBe('2 changes');
	});
});
