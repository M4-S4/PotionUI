<script lang="ts">
	import type { MediaToolModalProps } from '$lib/tools/tools';
	import HistoryEditImageModal from './HistoryEditImageModal.svelte';

	let { context, onClose, onDone, lookId }: MediaToolModalProps & { lookId?: string } = $props();
</script>

<HistoryEditImageModal
	{context}
	{onClose}
	{onDone}
	initialTool="filters"
	initialFilter={lookId}
/>
