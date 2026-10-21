<script lang="ts">
	import ConfirmDialog from '$lib/components/modals/ConfirmDialog.svelte';

	let { onConfirm }: { onConfirm: () => void } = $props();
	let key = $state(0);
</script>

<button data-testid="bump" onclick={() => key++}>bump</button>
<ConfirmDialog isOpen resetKey={key} confirmLabel="Go" hideCloseButton {onConfirm} onCancel={() => {}} />
