import type { LibrarySectionMeta } from '$lib/components/library/librarySection';
import type { User } from '$lib/stores/auth';
import type { UserGroup } from '$lib/services/admin-api';
import { CONTENT_POLICY_INHERIT, isContentPolicy } from '$lib/contentSafety/policy';

export const NEW_PLAN_ID = 'new';

export type UsersSubView = 'users' | 'groups' | 'plans';

export const USERS_LIBRARY_SECTIONS: readonly LibrarySectionMeta<UsersSubView>[] = [
	{ id: 'users', label: 'All users', icon: 'user' },
	{ id: 'groups', label: 'Groups', icon: 'group' },
	{ id: 'plans', label: 'Plans', icon: 'layers' }
];

export function usersSubViewFromParam(value: string | null): UsersSubView {
	return value === 'groups' || value === 'plans' ? value : 'users';
}

export interface AccountTypeCounts {
	ADMIN: number;
	USER: number;
}

export function accountTypeCounts(users: readonly User[]): AccountTypeCounts {
	return users.reduce<AccountTypeCounts>(
		(counts, user) => {
			if (user.account_type === 'ADMIN') counts.ADMIN += 1;
			else counts.USER += 1;
			return counts;
		},
		{ ADMIN: 0, USER: 0 }
	);
}

export interface DeletableGroupsResult {
	deletableIds: string[];
	skippedCount: number;
}

export function deletableGroupIds(groups: readonly UserGroup[], selectedIds: ReadonlySet<string>): DeletableGroupsResult {
	const selected = groups.filter((group) => selectedIds.has(group.id));
	const deletableIds = selected.filter((group) => !group.is_system).map((group) => group.id);
	return { deletableIds, skippedCount: selected.length - deletableIds.length };
}

export function deletableUserIds(selectedIds: ReadonlySet<string>, currentUserId: string | null | undefined): DeletableGroupsResult {
	const ids = Array.from(selectedIds);
	const deletableIds = ids.filter((id) => id !== currentUserId);
	return { deletableIds, skippedCount: ids.length - deletableIds.length };
}

export interface UserEditFormData {
	username: string;
	email: string;
	password: string;
	account_type: string;
}

export function userEditFormFrom(user: Pick<User, 'username' | 'email' | 'account_type'> | null): UserEditFormData {
	return user
		? { username: user.username, email: user.email, password: '', account_type: user.account_type }
		: { username: '', email: '', password: '', account_type: 'USER' };
}

export function userUpdatePayload(form: UserEditFormData): Partial<UserEditFormData> {
	const payload: Partial<UserEditFormData> = {};
	if (form.username) payload.username = form.username;
	if (form.email) payload.email = form.email;
	if (form.password) payload.password = form.password;
	if (form.account_type) payload.account_type = form.account_type;
	return payload;
}

export interface GroupEditFormData {
	name: string;
	description: string;
	content_policy: string;
}

export function groupEditFormFrom(
	group: { name: string; description?: string | null; content_policy?: unknown } | null
): GroupEditFormData {
	if (!group) return { name: '', description: '', content_policy: CONTENT_POLICY_INHERIT };
	return {
		name: group.name,
		description: group.description || '',
		content_policy: isContentPolicy(group.content_policy) ? group.content_policy : CONTENT_POLICY_INHERIT
	};
}

export function groupUpdatePayload(
	form: GroupEditFormData,
	policyLocked: boolean
): { name?: string; description: string; content_policy?: string | null } {
	const payload: { name?: string; description: string; content_policy?: string | null } = { description: form.description };
	if (form.name) payload.name = form.name;
	if (!policyLocked) payload.content_policy = isContentPolicy(form.content_policy) ? form.content_policy : null;
	return payload;
}

export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
	return `${count} ${count === 1 ? singular : plural}`;
}
