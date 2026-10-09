<script lang="ts">
	import { logger } from '$lib/utils/logger';
	import * as adminApi from '$lib/services/admin-api';
	import { toasts } from '$lib/stores/toast';
	import BaseModal from '$lib/components/modals/BaseModal.svelte';
	import ConfirmFooter from '$lib/components/modals/ConfirmFooter.svelte';
	import { createConfirmSettlementGate, getConfirmKeyboardAction, settleIfEligible } from '$lib/components/modals/confirmKeyboard';
	import { SegmentedControl } from '$lib/components/ui';
	import { CONTENT_POLICY_INHERIT, CONTENT_POLICY_OPTIONS, isContentPolicy } from '$lib/contentSafety/policy';
	import { useUsersDirectory } from './usersDirectory.svelte';

	let { onCreated }: { onCreated: (id: string | null) => void } = $props();

	const directory = useUsersDirectory();
	const gate = createConfirmSettlementGate();
	const contentPolicyItems = [{ id: CONTENT_POLICY_INHERIT, label: 'Inherit' }, ...CONTENT_POLICY_OPTIONS];

	let isOpen = $state(false);
	let saving = $state(false);
	let form = $state({ name: '', description: '', content_policy: CONTENT_POLICY_INHERIT });

	export function open() {
		form = { name: '', description: '', content_policy: CONTENT_POLICY_INHERIT };
		gate.reset();
		isOpen = true;
	}

	async function save() {
		saving = true;
		try {
			const response: any = await adminApi.createUserGroup({
				name: form.name,
				description: form.description,
				content_policy: isContentPolicy(form.content_policy) ? form.content_policy : null
			});
			await directory.loadGroups();
			await directory.loadMemberships();
			isOpen = false;
			onCreated(response?.data?.id ?? null);
		} catch (error) {
			logger.error('Failed to save group:', error);
			toasts.error('Failed to save group. Please check the form and try again.');
		} finally {
			saving = false;
			gate.reset();
		}
	}

	function cancel() {
		gate.settle(() => (isOpen = false));
	}

	function confirm() {
		settleIfEligible(gate, !!form.name && !saving, save);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (!isOpen) return;
		const { action, suppress } = getConfirmKeyboardAction(e);
		if (action === 'cancel') cancel();
		else if (action === 'confirm') confirm();
		if (suppress) e.preventDefault();
	}
</script>

<svelte:window onkeydowncapture={handleKeydown} />

<BaseModal {isOpen} title="Create New Group" size="md" handleEscapeKey={false} on:close={cancel}>
	<div class="p-6 space-y-4">
		<div>
			<label class="block text-sm font-medium text-fg-muted mb-2" for="group-name-input">Group Name</label>
			<input id="group-name-input" type="text" class="input w-full" bind:value={form.name} required />
		</div>
		<div>
			<label class="block text-sm font-medium text-fg-muted mb-2" for="group-description-input">Description</label>
			<textarea id="group-description-input" class="input w-full" rows="3" bind:value={form.description} placeholder="Optional description"></textarea>
		</div>
		<div>
			<span class="block text-sm font-medium text-fg-muted mb-2">Content policy</span>
			<SegmentedControl
				variant="toggle"
				ariaLabel="Group content policy"
				items={contentPolicyItems}
				selected={form.content_policy}
				onSelect={(id) => (form.content_policy = id)}
			/>
		</div>
	</div>
	<svelte:fragment slot="footer">
		<ConfirmFooter
			confirmLabel="Create Group"
			busy={saving}
			confirmDisabled={!form.name}
			onCancel={cancel}
			onConfirm={confirm}
		/>
	</svelte:fragment>
</BaseModal>
