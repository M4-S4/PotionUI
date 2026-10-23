<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import ConfirmDialog from '$lib/components/modals/ConfirmDialog.svelte';
	import { Input } from '$lib/components/ui';
	import { api } from '$lib/services/api';
	import { toasts } from '$lib/stores/toast';
	import type { Automation } from '$lib/types/automations';

	export let isOpen = false;

	const dispatch = createEventDispatcher<{ close: void; created: Automation }>();

	let name = '';
	let description = '';
	let creating = false;

	function resetAndClose() {
		name = '';
		description = '';
		dispatch('close');
	}

	function handleClose() {
		if (creating) return;
		resetAndClose();
	}

	async function handleCreate() {
		if (!name.trim()) return false;
		creating = true;
		try {
			const response = await api.createAutomation({
				name: name.trim(),
				description: description.trim() || undefined,
				graph: { nodes: [], edges: [] }
			});
			if (response.success && response.data) {
				dispatch('created', response.data);
				resetAndClose();
				return true;
			}
			toasts.error(response.error || 'Failed to create automation');
			return false;
		} catch {
			toasts.error('Failed to create automation');
			return false;
		} finally {
			creating = false;
		}
	}
</script>

<ConfirmDialog
	{isOpen}
	title="New automation"
	size="sm"
	busy={creating}
	confirmLabel="Create"
	confirmDisabled={!name.trim()}
	onConfirm={handleCreate}
	onCancel={handleClose}
>
	<div class="p-6 space-y-4">
		<div>
			<label for="automation-name" class="block text-xs font-medium text-fg-muted mb-1.5">Name</label>
			<Input id="automation-name" bind:value={name} placeholder="e.g. Index new LoRAs" />
		</div>
		<div>
			<label for="automation-description" class="block text-xs font-medium text-fg-muted mb-1.5">
				Description <span class="text-fg-subtle">(optional)</span>
			</label>
			<Input id="automation-description" bind:value={description} placeholder="What does this automation do?" />
		</div>
	</div>
</ConfirmDialog>
