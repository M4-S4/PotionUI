<script lang="ts">
	import MediaEditors from '$lib/media/editors/MediaEditors.svelte';
	import type { MediaEditorResult } from '$lib/media/editors/types';
	import { editorResultItems, landOnHistoryResult } from '$lib/tools/historyResult';
	import { applyLibraryPaintResult } from '$lib/tools/paintResult';
	import type { MediaToolModalProps } from '$lib/tools/tools';
	import { editImageRequest } from '../tools/editImage';

	let {
		context,
		onClose,
		initialTool,
		initialFilter
	}: MediaToolModalProps & { initialTool?: string; initialFilter?: string } = $props();

	let request = $derived.by(() => {
		const base = editImageRequest(context);
		if (!base || !initialTool) return base;
		return initialFilter ? { ...base, initialTool, initialFilter } : { ...base, initialTool };
	});

	async function handleResult(result: MediaEditorResult) {
		if (context.scope === 'library') {
			await applyLibraryPaintResult(
				result,
				context.items.map((item) => item.id)
			);
			return;
		}
		landed = editorResultItems(result);
	}

	let landed: ReturnType<typeof editorResultItems> | null = null;

	function handleClose() {
		const items = landed;
		landed = null;
		const first = context.items[0];
		const source = first?.generationId ? { generationId: first.generationId, fileIndex: first.paramIndex ?? 0 } : null;
		const startPath = window.location.pathname;
		onClose();
		if (!items || items.length === 0) return;
		setTimeout(() => landOnHistoryResult(items, { source, startPath }), 0);
	}
</script>

<MediaEditors {request} onClose={handleClose} onResult={handleResult} />
