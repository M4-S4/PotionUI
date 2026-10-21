<script lang="ts">
	import { untrack } from 'svelte';
	import ConfirmDialog from '$lib/components/modals/ConfirmDialog.svelte';
	import { isSubmitEnter } from '$lib/components/modals/confirmKeyboard';
	import { Input } from '$lib/components/ui';
	import { logger, getErrorMessage } from '$lib/utils/logger';
	import type { MutationResult } from './types';

	let {
		title,
		description,
		initial = '',
		onSubmit,
		onClose
	}: {
		title: string;
		description: string;
		initial?: string;
		onSubmit: (name: string) => Promise<MutationResult>;
		onClose: () => void;
	} = $props();

	const inputId = `collection-name-${Math.random().toString(36).slice(2, 10)}`;

	let name = $state(untrack(() => initial));
	let busy = $state(false);
	let error = $state<string | null>(null);
	let dialog: ReturnType<typeof ConfirmDialog> | undefined = $state();

	async function save(): Promise<boolean> {
		const trimmed = name.trim();
		if (!trimmed) {
			error = 'Enter a collection name.';
			return false;
		}
		busy = true;
		error = null;
		try {
			const result = await onSubmit(trimmed);
			if (result.success) {
				onClose();
				return true;
			}
			error = result.message ?? result.error ?? 'Could not save the collection.';
			return false;
		} catch (e) {
			logger.error('Collection save failed:', getErrorMessage(e));
			error = 'Could not save the collection.';
			return false;
		} finally {
			busy = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (!isSubmitEnter(e)) return;
		e.preventDefault();
		dialog?.confirm();
	}
</script>

<ConfirmDialog
	bind:this={dialog}
	isOpen={true}
	size="sm"
	{title}
	subtitle={description}
	{busy}
	confirmLabel="Save"
	confirmDisabled={!name.trim()}
	onConfirm={save}
	onCancel={onClose}
>
	<div class="px-6 py-5">
		<label class="mb-2 block text-sm text-fg-muted" for={inputId}>Collection name</label>
		<Input
			id={inputId}
			bind:value={name}
			invalid={!!error}
			class="text-sm"
			autocomplete="off"
			data-autofocus
			oninput={() => (error = null)}
			onkeydown={handleKeydown}
		/>
		{#if error}
			<p class="mt-2 text-sm text-danger" role="alert">{error}</p>
		{/if}
	</div>
</ConfirmDialog>
