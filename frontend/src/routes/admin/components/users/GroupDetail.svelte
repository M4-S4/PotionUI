<script lang="ts">
	import { untrack } from 'svelte';
	import { logger } from '$lib/utils/logger';
	import * as adminApi from '$lib/services/admin-api';
	import * as plansApi from '$lib/plans/api';
	import type { GroupImpact } from '$lib/plans/types';
	import { toasts } from '$lib/stores/toast';
	import { confirmDialog } from '$lib/stores/confirm';
	import { AssignedList, type PickerDiff } from '$lib/components/picker';
	import { presetsKind, llmsKind, usersKind, modelsKind, type PickerModel } from '$lib/components/picker/kinds';
	import { Badge, Input, IconButton, SegmentedControl } from '$lib/components/ui';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import { CONTENT_POLICY_INHERIT, CONTENT_POLICY_OPTIONS, isSystemRestrictedGroup } from '$lib/contentSafety/policy';
	import { DetailHeader, DetailTabs, DetailBody, DetailLayout, DetailSection, DetailFooter, DetailField } from '$lib/components/detail';
	import { adminSectionIcon } from '../../adminSections';
	import GroupPlanSection from '../plans/GroupPlanSection.svelte';
	import { createModelRemote, fetchAssignedModels } from './modelAssignmentSource';
	import { applyAssignments, ensureOk } from './assignments';
	import { useUsersDirectory } from './usersDirectory.svelte';
	import { groupEditFormFrom, groupUpdatePayload } from './usersLibrary';

	type Tab = 'overview' | 'users' | 'presets' | 'llms' | 'models';

	let { group, onBack }: { group: adminApi.UserGroup; onBack: () => void } = $props();

	const directory = useUsersDirectory();
	const contentPolicyItems = [{ id: CONTENT_POLICY_INHERIT, label: 'Inherit' }, ...CONTENT_POLICY_OPTIONS];

	let tab = $state<Tab>('overview');

	const initialForm = untrack(() => groupEditFormFrom(group));
	let form = $state(initialForm);
	let snapshot = $state(JSON.stringify(initialForm));
	let saving = $state(false);
	const initialPlan = untrack(() => directory.groupPlanById[group.id] ?? null);
	let planSaved = $state<string | null>(initialPlan);
	let planDraft = $state<string | null>(initialPlan);
	let impact = $state<GroupImpact | null>(null);
	let loadingImpact = $state(false);
	const dirty = $derived(JSON.stringify(form) !== snapshot || planDraft !== planSaved);

	let llms = $state<any[] | null>(null);
	let presets = $state<any[] | null>(null);
	let modelRows = $state<PickerModel[] | null>(null);
	let loadingLLMs = $state(false);
	let loadingPresets = $state(false);

	const restricted = $derived(isSystemRestrictedGroup(group));
	const userRows = $derived(directory.pickerUsers.filter((u) => (directory.userGroupIds[u.id] || []).includes(group.id)));
	const presetRows = $derived(directory.allPresets.filter((p) => (presets ?? []).some((gp) => gp.preset_id === p.preset_db_id)));
	const llmRows = $derived(directory.allLLMConfigs.filter((l) => (llms ?? []).some((gl) => gl.llm_config_id === l.id)));
	const modelRemote = $derived(createModelRemote({ groupId: group.id }, directory.modelTypeOptions));

	const tabs = $derived([
		{ id: 'overview', label: 'Overview', icon: 'info' },
		{ id: 'users', label: 'Users', icon: 'user', count: group.member_count ?? 0 },
		{ id: 'presets', label: 'Presets', icon: 'layers', count: presets?.length ?? 0 },
		{ id: 'llms', label: 'LLMs', icon: 'chat', count: llms?.length ?? 0 },
		{ id: 'models', label: 'Models', icon: 'cube', count: modelRows?.length ?? 0 }
	]);

	$effect(() => {
		const planId = planDraft;
		if (tab !== 'overview') return;
		untrack(() => void loadImpact(planId));
	});

	function selectTab(id: string) {
		tab = id as Tab;
		loadTabData(tab);
	}

	function loadTabData(next: Tab) {
		if (next === 'presets' && !presets && !loadingPresets) void loadPresets();
		else if (next === 'llms' && !llms && !loadingLLMs) void loadLLMs();
		else if (next === 'models') {
			void directory.ensureModelTypeOptions();
			if (!modelRows) void loadModelAssignments();
		}
	}

	async function loadImpact(planId: string | null) {
		loadingImpact = true;
		try {
			const result = await plansApi.getGroupPlanImpact(group.id, planId);
			if (planDraft === planId) impact = result;
		} catch (error) {
			logger.error('Failed to load plan impact:', error);
			impact = null;
		} finally {
			loadingImpact = false;
		}
	}

	async function loadLLMs() {
		loadingLLMs = true;
		try {
			const result = await adminApi.getGroupLLMs(group.id);
			if (result.success) llms = result.data || [];
		} catch (error) {
			logger.error('Failed to load group LLMs:', error);
		} finally {
			loadingLLMs = false;
		}
	}

	async function loadPresets() {
		loadingPresets = true;
		try {
			const result = await adminApi.getGroupPresets(group.id);
			if (result.success) presets = result.data || [];
		} catch (error) {
			logger.error('Failed to load group presets:', error);
		} finally {
			loadingPresets = false;
		}
	}

	async function loadModelAssignments() {
		try {
			modelRows = await fetchAssignedModels({ groupId: group.id });
		} catch (error) {
			logger.error('Failed to load group model assignments:', error);
			modelRows = [];
			toasts.error('Failed to load assigned models.');
		}
	}

	function applyUsers(diff: PickerDiff) {
		return applyAssignments(
			diff,
			{
				assign: async (userId) => ensureOk(await adminApi.addUsersToGroup(group.id, [userId]), 'Could not add user'),
				unassign: async (userId) => ensureOk(await adminApi.removeUserFromGroup(group.id, userId), 'Could not remove user')
			},
			() => directory.refreshGroupMembers(group.id),
			'group members'
		);
	}

	function applyPresets(diff: PickerDiff) {
		return applyAssignments(
			diff,
			{
				assign: async (id) => ensureOk(await adminApi.assignPresetsToGroup(group.id, [id]), 'Could not assign preset'),
				unassign: async (id) => ensureOk(await adminApi.unassignPresetFromGroup(group.id, id), 'Could not unassign preset')
			},
			loadPresets,
			'presets'
		);
	}

	function applyLLMs(diff: PickerDiff) {
		return applyAssignments(
			diff,
			{
				assign: async (id) => ensureOk(await adminApi.assignLLMsToGroup(group.id, [id]), 'Could not assign LLM'),
				unassign: async (id) => ensureOk(await adminApi.unassignLLMFromGroup(group.id, id), 'Could not unassign LLM')
			},
			loadLLMs,
			'LLMs'
		);
	}

	function applyModels(diff: PickerDiff) {
		return applyAssignments(
			diff,
			{
				assign: async (id) => ensureOk(await adminApi.assignModelsToGroup(group.id, [id]), 'Could not assign model'),
				unassign: async (id) => ensureOk(await adminApi.unassignModelFromGroup(group.id, id), 'Could not unassign model')
			},
			loadModelAssignments,
			'models'
		);
	}

	function loadForm(source: adminApi.UserGroup | null) {
		form = groupEditFormFrom(source);
		snapshot = JSON.stringify(form);
	}

	function discard() {
		loadForm(group);
		planDraft = planSaved;
	}

	async function save() {
		saving = true;
		try {
			const response: any = await adminApi.updateUserGroup(group.id, groupUpdatePayload(form, restricted));
			if (response.success !== false) {
				if (planDraft !== planSaved) {
					await plansApi.setGroupPlan(group.id, planDraft);
					planSaved = planDraft;
					await directory.loadPlans();
				}
				toasts.success(`${form.name || group.name} updated`);
				await directory.loadGroups();
				loadForm(directory.groups.find((g) => g.id === group.id) ?? null);
			} else {
				toasts.error(response.message || 'Failed to save group');
			}
		} catch (error) {
			logger.error('Failed to save group:', error);
			toasts.error('Failed to save group. Please check the form and try again.');
		} finally {
			saving = false;
		}
	}

	async function remove() {
		if (group.is_system) return;
		if (
			await confirmDialog({
				title: `Are you sure you want to delete group "${group.name}"?`,
				message: 'This action cannot be undone.',
				variant: 'danger'
			})
		) {
			try {
				await adminApi.deleteUserGroup(group.id);
				await directory.loadGroups();
				await directory.loadMemberships();
				onBack();
			} catch (error: any) {
				logger.error('Failed to delete group:', error);
				toasts.error(error?.response?.data?.detail || 'Failed to delete group.');
			}
		}
	}
</script>

<div class="flex h-full flex-col">
	<DetailHeader title={group.name} icon={adminSectionIcon('users')} backLabel="Users" {onBack}>
		{#snippet chips()}
			{#if group.is_system}<Badge variant="neutral" size="sm">Built-in</Badge>{/if}
			{#if restricted}<Badge variant="danger" size="sm">Restricted</Badge>{/if}
		{/snippet}
		{#snippet subtitle()}
			{group.member_count ?? 0} members
		{/snippet}
		{#snippet actions()}
			<Tooltip text={group.is_system ? "Built-in group — can't be deleted" : 'Delete group'}>
				<IconButton
					icon="trash"
					label="Delete group"
					class="text-danger hover:text-danger hover:bg-danger/10"
					disabled={group.is_system}
					onclick={remove}
				/>
			</Tooltip>
		{/snippet}
	</DetailHeader>

	<DetailTabs tabs={tabs} active={tab} onSelect={selectTab} ariaLabel="Group details" />

	{#if tab === 'overview'}
		<DetailBody>
			<DetailLayout>
				{#snippet main()}
					<DetailSection label="Identity">
						<div class="space-y-4">
							<DetailField label="Group Name" id="edit-group-name">
								<Input id="edit-group-name" type="text" bind:value={form.name} />
							</DetailField>
							<DetailField label="Description" id="edit-group-description" wide>
								<textarea id="edit-group-description" class="input" rows="3" bind:value={form.description} placeholder="Optional description"></textarea>
							</DetailField>
						</div>
					</DetailSection>
					<DetailSection label="Content policy">
						{#if restricted}
							<p class="text-sm text-fg-muted" data-restricted-group-note>
								Members are always Blocked. They cannot generate or see NSFW content, cannot change their
								own content mode, and see only media that has been rated. This cannot be overridden.
							</p>
						{:else}
							<div class="flex items-start justify-between gap-6">
								<p class="text-sm text-fg-muted">
									Applies to members of this group. Inherit uses the instance policy.
								</p>
								<SegmentedControl
									variant="toggle"
									ariaLabel="Group content policy"
									items={contentPolicyItems}
									selected={form.content_policy}
									onSelect={(id) => (form.content_policy = id)}
								/>
							</div>
						{/if}
					</DetailSection>
					<GroupPlanSection
						plans={directory.plans}
						kinds={directory.planKinds}
						planId={planDraft}
						defaultPlanName={directory.defaultPlanName}
						{impact}
						{loadingImpact}
						onChange={(planId) => (planDraft = planId)}
					/>
				{/snippet}
			</DetailLayout>
		</DetailBody>
	{:else if tab === 'users'}
		<DetailBody>
			<DetailLayout>
				{#snippet main()}
					<AssignedList
						kind={usersKind}
						label="Users"
						assignedRows={userRows}
						pickerItems={directory.pickerUsers}
						onApply={applyUsers}
						loading={directory.loadingMemberships}
						pickerSubtitle={`Add users to ${group.name}`}
					/>
				{/snippet}
			</DetailLayout>
		</DetailBody>
	{:else if tab === 'presets'}
		<DetailBody>
			<DetailLayout>
				{#snippet main()}
					<AssignedList
						kind={presetsKind}
						label="Presets"
						assignedRows={presetRows}
						pickerItems={directory.allPresets}
						onApply={applyPresets}
						loading={!presets && loadingPresets}
						pickerSubtitle={`Grant ${group.name} access to presets`}
					/>
				{/snippet}
			</DetailLayout>
		</DetailBody>
	{:else if tab === 'llms'}
		<DetailBody>
			<DetailLayout>
				{#snippet main()}
					<AssignedList
						kind={llmsKind}
						label="LLM configurations"
						assignedRows={llmRows}
						pickerItems={directory.allLLMConfigs}
						onApply={applyLLMs}
						loading={!llms && loadingLLMs}
						pickerSubtitle={`Grant ${group.name} access to LLM configurations`}
					/>
				{/snippet}
			</DetailLayout>
		</DetailBody>
	{:else if tab === 'models'}
		<DetailBody>
			<DetailLayout>
				{#snippet main()}
					<AssignedList
						kind={modelsKind}
						label="Models"
						assignedRows={modelRows ?? []}
						remote={modelRemote}
						onApply={applyModels}
						loading={!modelRows}
						pickerSubtitle={`Grant ${group.name} access to models`}
					/>
				{/snippet}
			</DetailLayout>
		</DetailBody>
	{/if}

	{#if tab === 'overview'}
		<DetailFooter dirtyCount={dirty ? 1 : 0} saving={saving} onSave={save} onDiscard={discard} />
	{/if}
</div>
