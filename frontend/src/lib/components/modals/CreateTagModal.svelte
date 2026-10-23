<script lang="ts">
	import { logger } from '$lib/utils/logger';
	import ConfirmDialog from './ConfirmDialog.svelte';
	import Icon from '$lib/components/Icon.svelte';

	export let onClose: () => void;
	export let onCreate: (name: string) => Promise<{ success: boolean }>;
	export let description: string;

	let newTagName = '';
	let creatingTag = false;
	let dialog: ReturnType<typeof ConfirmDialog>;

	async function handleCreateTag() {
		if (!newTagName.trim() || creatingTag) return false;

		try {
			creatingTag = true;
			const response = await onCreate(newTagName.trim());
			if (response.success) {
				newTagName = '';
				onClose();
				return true;
			}
			return false;
		} catch (error) {
			logger.error('Failed to create tag:', error);
			return false;
		} finally {
			creatingTag = false;
		}
	}

	function handleCancel() {
		newTagName = '';
		onClose();
	}

	function handleInputKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') dialog.confirm();
	}
</script>

<ConfirmDialog
	bind:this={dialog}
	isOpen={true}
	size="sm"
	hideCloseButton
	busy={creatingTag}
	labelledBy="add-tag-title"
	confirmLabel="Confirm"
	confirmDisabled={!newTagName.trim() || creatingTag}
	onConfirm={handleCreateTag}
	onCancel={handleCancel}
>
	<div class="p-6">
		<div class="flex items-center gap-2 mb-4">
			<Icon name="tag" className="w-5 h-5 text-fg" />
			<h3 id="add-tag-title" class="text-lg font-semibold text-fg">Create New Tag</h3>
		</div>
		<div class="mb-4">
			<label class="block text-sm text-fg-muted mb-2" for="new-tag-name">Tag Name</label>
			<input
				id="new-tag-name"
				type="text"
				class="w-full px-3 py-2 bg-surface-2 border border-line-strong text-fg rounded-lg focus:outline-none focus:ring-2 focus:ring-accent"
				placeholder="Enter tag name..."
				bind:value={newTagName}
				on:keydown={handleInputKeydown}
			/>
			<p class="text-sm text-fg-muted mt-2">{description}</p>
		</div>
	</div>
</ConfirmDialog>
