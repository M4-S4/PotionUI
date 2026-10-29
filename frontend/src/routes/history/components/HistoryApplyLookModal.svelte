<script lang="ts">
	import { FILTERS_TOOL, lookTool } from '$lib/components/imageEditor/initialTool';
	import type { MediaToolModalProps } from '$lib/tools/tools';
	import HistoryEditImageModal from './HistoryEditImageModal.svelte';

	let { context, onClose, onDone, lookId }: MediaToolModalProps & { lookId?: string } = $props();
</script>

<HistoryEditImageModal
	{context}
	{onClose}
	{onDone}
	initialTool={lookId ? lookTool(lookId) : FILTERS_TOOL}
/>
