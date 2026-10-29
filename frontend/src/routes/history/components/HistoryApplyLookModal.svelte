<script lang="ts">
	import type { MediaToolModalProps } from '$lib/tools/tools';
	import HistoryEditImageModal from './HistoryEditImageModal.svelte';

	let { context, onClose, onDone }: MediaToolModalProps = $props();
</script>

<HistoryEditImageModal {context} {onClose} {onDone} initialTool="filters" />
