<script lang="ts">
	import { logger } from '$lib/utils/logger';
	import * as adminApi from '$lib/services/admin-api';
	import { toasts } from '$lib/stores/toast';
	import BaseModal from '$lib/components/modals/BaseModal.svelte';
	import ConfirmFooter from '$lib/components/modals/ConfirmFooter.svelte';
	import { createConfirmSettlementGate, getConfirmKeyboardAction, settleIfEligible } from '$lib/components/modals/confirmKeyboard';
	import { useUsersDirectory } from './usersDirectory.svelte';

	let { onCreated }: { onCreated: (id: string | null) => void } = $props();

	const directory = useUsersDirectory();
	const gate = createConfirmSettlementGate();

	let isOpen = $state(false);
	let saving = $state(false);
	let form = $state({ username: '', email: '', password: '', account_type: 'USER' });

	export function open() {
		form = { username: '', email: '', password: '', account_type: 'USER' };
		gate.reset();
		isOpen = true;
	}

	async function save() {
		saving = true;
		try {
			const response: any = await adminApi.createUser(form);
			await directory.loadUsers();
			isOpen = false;
			onCreated(response?.data?.id ?? null);
		} catch (error) {
			logger.error('Failed to save user:', error);
			toasts.error('Failed to save user. Please check the form and try again.');
		} finally {
			saving = false;
			gate.reset();
		}
	}

	function cancel() {
		gate.settle(() => (isOpen = false));
	}

	function confirm() {
		settleIfEligible(gate, !!form.username && !!form.email && !!form.password && !saving, save);
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

<BaseModal {isOpen} title="Create New User" size="md" handleEscapeKey={false} on:close={cancel}>
	<div class="p-6 space-y-4">
		<div>
			<label class="block text-sm font-medium text-fg-muted mb-2" for="user-username-input">Username</label>
			<input id="user-username-input" type="text" class="input w-full" bind:value={form.username} required />
		</div>
		<div>
			<label class="block text-sm font-medium text-fg-muted mb-2" for="user-email-input">Email</label>
			<input id="user-email-input" type="email" class="input w-full" bind:value={form.email} required />
		</div>
		<div>
			<label class="block text-sm font-medium text-fg-muted mb-2" for="user-password-input">Password</label>
			<input id="user-password-input" type="password" class="input w-full" bind:value={form.password} required />
		</div>
		<div>
			<label class="block text-sm font-medium text-fg-muted mb-2" for="user-type-input">Account Type</label>
			<select id="user-type-input" class="input w-full" bind:value={form.account_type}>
				<option value="USER">Regular User</option>
				<option value="ADMIN">Administrator</option>
			</select>
		</div>
	</div>
	<svelte:fragment slot="footer">
		<ConfirmFooter
			confirmLabel="Create User"
			busy={saving}
			confirmDisabled={!form.username || !form.email || !form.password}
			onCancel={cancel}
			onConfirm={confirm}
		/>
	</svelte:fragment>
</BaseModal>
