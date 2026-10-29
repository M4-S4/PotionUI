<script lang="ts">
	import { setContext } from 'svelte';
	import { PAINT_INITIAL_TOOL } from '$lib/components/imageEditor/initialTool';
	import type { MediaToolModalProps } from '$lib/tools/tools';
	import HistoryEditImageModal from './HistoryEditImageModal.svelte';

	let { context, onClose, onDone }: MediaToolModalProps = $props();

	setContext(PAINT_INITIAL_TOOL, () => 'filters');
</script>

<HistoryEditImageModal {context} {onClose} {onDone} />
