<script lang="ts">
	import type { GenerationHistoryItem } from '$lib/types/history';
	import ConfirmModal from '$lib/components/modals/ConfirmModal.svelte';
	import { deleteCopy } from '$lib/utils/historyDeleteCopy';

	export let generation: GenerationHistoryItem;
	export let onCancel: () => void;
	export let onConfirm: () => void;

	// Remounts ConfirmModal (fresh settlement gate) per attempt so a failed
	// delete that leaves the modal open can still be retried via keyboard.
	let attempt = 0;
	function handleConfirm() {
		attempt += 1;
		onConfirm();
	}

	$: copy = deleteCopy(generation);
</script>

{#key attempt}
	<ConfirmModal
		isOpen={true}
		title={copy.title}
		message={copy.message}
		variant="danger"
		confirmLabel="Delete"
		on:confirm={handleConfirm}
		on:cancel={onCancel}
	/>
{/key}
