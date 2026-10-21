<script lang="ts" generics="T extends CollectionLike">
	import { untrack } from 'svelte';
	import ConfirmDialog from '$lib/components/modals/ConfirmDialog.svelte';
	import { buildTree, flattenTree } from '$lib/components/pane';
	import { logger, getErrorMessage } from '$lib/utils/logger';
	import type { CollectionLike, MutationResult } from './types';

	const TOP_LEVEL = '';

	let {
		title,
		collections,
		blockedIds,
		showTopLevel,
		onMove,
		onClose
	}: {
		title: string;
		collections: T[];
		blockedIds: Set<string>;
		showTopLevel: boolean;
		onMove: (targetId: string | null) => Promise<MutationResult>;
		onClose: () => void;
	} = $props();

	const selectId = `collection-move-${Math.random().toString(36).slice(2, 10)}`;

	let targets = $derived(
		flattenTree(buildTree(collections)).filter((node) => !blockedIds.has(node.item.id))
	);
	let target = $state(
		untrack(() => (showTopLevel ? TOP_LEVEL : (targets[0]?.item.id ?? TOP_LEVEL)))
	);
	let hasDestination = $derived(showTopLevel || targets.length > 0);
	let busy = $state(false);
	let error = $state<string | null>(null);

	async function move(): Promise<boolean> {
		busy = true;
		error = null;
		try {
			const result = await onMove(target === TOP_LEVEL ? null : target);
			if (result.success) {
				onClose();
				return true;
			}
			error = result.error ?? result.message ?? 'Move failed.';
			return false;
		} catch (e) {
			logger.error('Move failed:', getErrorMessage(e));
			error = 'Move failed. Choose a parent outside the moved collections.';
			return false;
		} finally {
			busy = false;
		}
	}
</script>

<ConfirmDialog
	isOpen={true}
	size="sm"
	{title}
	subtitle="Choose a parent collection."
	{busy}
	confirmLabel="Move"
	confirmDisabled={!hasDestination}
	onConfirm={move}
	onCancel={onClose}
>
	<div class="px-6 py-5">
		{#if hasDestination}
			<label class="mb-2 block text-sm text-fg-muted" for={selectId}>Destination</label>
			<select id={selectId} class="input text-sm" bind:value={target} data-autofocus>
				{#if showTopLevel}
					<option value={TOP_LEVEL}>Top level</option>
				{/if}
				{#each targets as node (node.item.id)}
					<option value={node.item.id}>{'   '.repeat(node.depth)}{node.item.name}</option>
				{/each}
			</select>
		{:else}
			<p class="text-sm text-fg-muted">No other collection can hold this one.</p>
		{/if}
		{#if error}
			<p class="mt-2 text-sm text-danger" role="alert">{error}</p>
		{/if}
	</div>
</ConfirmDialog>
