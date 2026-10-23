<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import ConfirmDialog from '$lib/components/modals/ConfirmDialog.svelte';
	import { isSubmitEnter } from '$lib/components/modals/confirmKeyboard';
	import { Input, Alert } from '$lib/components/ui';

	export let isOpen: boolean = false;
	/** 'rename' updates the current session in place; 'save-as' creates a new one. */
	export let mode: 'rename' | 'save-as' = 'rename';
	export let sessionName: string = '';
	export let nameError: string = '';
	export let error: string | null = null;
	export let isSaving: boolean = false;

	const dispatch = createEventDispatcher<{ close: void; confirm: void }>();
	let dialog: ReturnType<typeof ConfirmDialog>;

	$: title = mode === 'rename' ? 'Rename Session' : 'Save New Session';
	$: confirmLabel = mode === 'rename' ? 'Update' : 'Save';

	function handleNameKeydown(event: KeyboardEvent) {
		if (!isOpen || isSaving || !isSubmitEnter(event)) return;
		event.preventDefault();
		dialog.confirm();
	}
</script>

<ConfirmDialog
	bind:this={dialog}
	{isOpen}
	{title}
	size="sm"
	busy={isSaving}
	resetKey={isSaving}
	{confirmLabel}
	confirmDisabled={!sessionName.trim()}
	onConfirm={() => dispatch('confirm')}
	onCancel={() => dispatch('close')}
>
	<div class="space-y-4 p-4 sm:p-6">
		<div>
			<label class="mb-1.5 block text-sm font-medium text-fg" for="session-save-name">Name</label>
			<Input
				id="session-save-name"
				type="text"
				bind:value={sessionName}
				placeholder="Enter session name"
				invalid={!!nameError}
				onkeydown={handleNameKeydown}
				data-autofocus
			/>
			{#if nameError}
				<p class="mt-1.5 text-xs text-danger">{nameError}</p>
			{/if}
		</div>

		{#if error}
			<Alert variant="danger">{error}</Alert>
		{/if}
	</div>
</ConfirmDialog>
