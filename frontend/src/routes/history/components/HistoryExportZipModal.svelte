<script lang="ts">
	import ConfirmDialog from '$lib/components/modals/ConfirmDialog.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import { api } from '$lib/services/api/index';
	import { toasts } from '$lib/stores/toast';
	import { logger, getErrorMessage } from '$lib/utils/logger';
	import type { MediaToolModalProps } from '$lib/tools/tools';

	// Downloading reads the selection, it does not change it, so a finished
	// export closes rather than settling the tool as "data changed".
	let { context, onClose }: MediaToolModalProps = $props();

	let stripMetadata = $state(false);
	let exporting = $state(false);

	let count = $derived(context.items.length);

	async function download() {
		exporting = true;
		try {
			if (context.scope === 'library') {
				await api.exportLibraryItems(context.items.map((item) => item.id));
			} else {
				await api.exportGenerations(context.generationIds, stripMetadata);
			}
			toasts.success('Export started');
			onClose();
			return true;
		} catch (error) {
			logger.error('Export failed:', getErrorMessage(error));
			toasts.error('Export failed');
			return false;
		} finally {
			exporting = false;
		}
	}
</script>

<ConfirmDialog
	isOpen={true}
	title="Download .zip"
	size="md"
	busy={exporting}
	resetKey={context}
	confirmLabel="Download"
	confirmDisabled={count === 0}
	onConfirm={download}
	onCancel={onClose}
>
	{#snippet headerIconContent()}
		<Icon name="download" className="h-5 w-5 text-fg-muted" />
	{/snippet}

	<div class="space-y-4 p-4 sm:p-6">
		<p class="text-sm text-fg-muted tabular-nums">
			{count} {context.scope === 'library' ? 'item' : 'generation'}{count === 1 ? '' : 's'} in the archive.
		</p>

		{#if context.scope === 'history'}
			<label class="flex items-center gap-2 text-sm text-fg-muted cursor-pointer">
				<input type="checkbox" bind:checked={stripMetadata} class="accent-accent" />
				Strip metadata
			</label>
		{/if}
	</div>
</ConfirmDialog>
