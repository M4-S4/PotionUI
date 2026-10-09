import { getContext, setContext } from 'svelte';
import { logger, getApiErrorMessage } from '$lib/utils/logger';
import * as adminApi from '$lib/services/admin-api';
import { api } from '$lib/services/api/index';
import { toasts } from '$lib/stores/toast';
import type { User } from '$lib/stores/auth';
import { isSystemRestrictedGroup } from '$lib/contentSafety/policy';
import type { PickerUser } from '$lib/components/picker/kinds';
import * as plansApi from '$lib/plans/api';
import type { LimitKindDescriptor, Plan, PlansSettings, UserUsageRow } from '$lib/plans/types';
import { sourceLabel } from '$lib/plans/usage';
import { fetchModelTypeOptions } from './modelAssignmentSource';

const EMPTY_PLANS_SETTINGS: PlansSettings = { default_plan_id: null, exempt_admins: true, day_timezone: 'UTC', contact_line: '' };

export class UsersDirectory {
	users = $state<User[]>([]);
	loadingUsers = $state(true);
	usersError = $state<string | null>(null);

	groups = $state<adminApi.UserGroup[]>([]);
	loadingGroups = $state(true);
	groupsError = $state<string | null>(null);

	membersByGroup = $state<Record<string, adminApi.UserGroupMember[]>>({});
	loadingMemberships = $state(true);

	allLLMConfigs = $state<any[]>([]);
	allPresets = $state<any[]>([]);
	modelTypeOptions = $state<{ value: string; label: string }[]>([]);

	plans = $state<Plan[]>([]);
	planKinds = $state<LimitKindDescriptor[]>([]);
	planKindsInUse = $state<LimitKindDescriptor[]>([]);
	plansSettings = $state<PlansSettings>(EMPTY_PLANS_SETTINGS);
	usageByUser = $state<Map<string, UserUsageRow>>(new Map());
	groupPlanById = $state<Record<string, string | null>>({});
	loadingPlans = $state(true);
	plansError = $state<string | null>(null);

	restrictedGroupIds = $derived(this.groups.filter((g) => isSystemRestrictedGroup(g)).map((g) => g.id));

	userGroupIds = $derived.by(() => {
		const map: Record<string, string[]> = {};
		for (const [groupId, members] of Object.entries(this.membersByGroup)) {
			for (const member of members) {
				(map[member.user_id] ||= []).push(groupId);
			}
		}
		return map;
	});

	defaultPlanName = $derived(this.plans.find((p) => p.id === this.plansSettings.default_plan_id)?.name ?? null);

	pickerUsers = $derived<PickerUser[]>(
		this.users.map((u) => {
			const usage = this.usageByUser.get(u.id);
			const groupNames = (this.userGroupIds[u.id] || [])
				.map((id) => this.groups.find((g) => g.id === id)?.name)
				.filter((name): name is string => !!name);
			return {
				...u,
				groups: groupNames,
				plan: usage?.plan?.name ?? null,
				planSource: usage ? sourceLabel(usage.source, usage.exempt) : null
			};
		})
	);

	private modelTypeOptionsRequested = false;

	isUserRestricted(userId: string): boolean {
		return (this.userGroupIds[userId] || []).some((id) => this.restrictedGroupIds.includes(id));
	}

	async bootstrap() {
		await Promise.all([this.loadUsers(), this.loadGroups(), this.loadPlans()]);
		await this.loadMemberships();
		await Promise.all([this.loadLLMConfigs(), this.loadPresets()]);
	}

	async loadUsers() {
		try {
			this.loadingUsers = true;
			const response = await adminApi.getUsers();
			if (response.success && response.data) {
				this.users = response.data;
				this.usersError = null;
			} else {
				this.usersError = response.message || 'Failed to load users';
			}
		} catch (error) {
			logger.error('Failed to load users:', error);
			this.usersError = getApiErrorMessage(error, 'Failed to load users');
		} finally {
			this.loadingUsers = false;
		}
	}

	async loadGroups() {
		try {
			this.loadingGroups = true;
			const response = await adminApi.getUserGroups();
			if (response.success) {
				this.groups = response.data || [];
				this.groupsError = null;
			} else {
				this.groupsError = response.message || 'Failed to load user groups';
			}
		} catch (error) {
			logger.error('Failed to load user groups:', error);
			this.groupsError = getApiErrorMessage(error, 'Failed to load user groups');
		} finally {
			this.loadingGroups = false;
		}
	}

	async loadMemberships() {
		if (this.groups.length === 0) {
			this.membersByGroup = {};
			this.loadingMemberships = false;
			return;
		}
		this.loadingMemberships = true;
		try {
			const results = await Promise.all(this.groups.map((g) => adminApi.getGroupMembers(g.id)));
			const map: Record<string, adminApi.UserGroupMember[]> = {};
			this.groups.forEach((g, i) => {
				map[g.id] = results[i].success ? (results[i].data as any[]) || [] : [];
			});
			this.membersByGroup = map;
		} catch (error) {
			logger.error('Failed to load group memberships:', error);
		} finally {
			this.loadingMemberships = false;
		}
	}

	async refreshGroupMembers(groupId: string) {
		const membersResult = await adminApi.getGroupMembers(groupId);
		if (!membersResult.success) return;
		const members = membersResult.data || [];
		this.membersByGroup = { ...this.membersByGroup, [groupId]: members };
		this.groups = this.groups.map((g) => (g.id === groupId ? { ...g, member_count: members.length } : g));
	}

	async loadLLMConfigs() {
		try {
			const response = await api.getLLMConfigurations();
			if (response.success && response.data) {
				this.allLLMConfigs = response.data.configurations || [];
			}
		} catch (error) {
			logger.error('Failed to load LLM configurations:', error);
		}
	}

	async loadPresets() {
		try {
			const response = await api.listPresets(true);
			if (response.success && response.data) {
				this.allPresets = response.data.filter((preset: any) => preset.installed);
			}
		} catch (error) {
			logger.error('Failed to load presets:', error);
		}
	}

	async ensureModelTypeOptions() {
		if (this.modelTypeOptions.length > 0 || this.modelTypeOptionsRequested) return;
		this.modelTypeOptionsRequested = true;
		try {
			this.modelTypeOptions = await fetchModelTypeOptions();
		} finally {
			this.modelTypeOptionsRequested = false;
		}
	}

	async loadPlans() {
		this.loadingPlans = true;
		try {
			const [overview, usage, groupRows] = await Promise.all([
				plansApi.getPlansOverview(),
				plansApi.listUsersUsage(),
				plansApi.listGroupPlans()
			]);
			this.planKinds = overview.kinds;
			this.plans = overview.plans;
			this.plansSettings = overview.settings;
			this.planKindsInUse = usage.kinds;
			this.usageByUser = new Map(usage.users.map((row) => [row.user_id, row]));
			this.groupPlanById = Object.fromEntries(groupRows.map((row) => [row.id, row.plan?.id ?? null]));
			this.plansError = null;
		} catch (error) {
			logger.error('Failed to load plans:', error);
			this.plansError = getApiErrorMessage(error, 'Failed to load plans');
		} finally {
			this.loadingPlans = false;
		}
	}

	async savePlansSettings(patch: Partial<PlansSettings>) {
		const previous = this.plansSettings;
		this.plansSettings = { ...this.plansSettings, ...patch };
		try {
			this.plansSettings = await plansApi.updatePlansSettings(patch);
			if ('default_plan_id' in patch) await this.loadPlans();
		} catch (error) {
			this.plansSettings = previous;
			logger.error('Failed to save plan settings:', error);
			toasts.error(getApiErrorMessage(error, 'Failed to save plan settings.'));
		}
	}
}

const CONTEXT_KEY = Symbol('users-directory');

export function provideUsersDirectory(): UsersDirectory {
	const directory = new UsersDirectory();
	setContext(CONTEXT_KEY, directory);
	return directory;
}

export function useUsersDirectory(): UsersDirectory {
	return getContext<UsersDirectory>(CONTEXT_KEY);
}
