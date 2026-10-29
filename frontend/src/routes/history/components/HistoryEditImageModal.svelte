<script lang="ts">
	import MediaEditors from '$lib/media/editors/MediaEditors.svelte';
	import type { MediaEditorResult } from '$lib/media/editors/types';
	import { toasts } from '$lib/stores/toast';
	import { applyLibraryPaintResult } from '$lib/tools/paintResult';
	import type { MediaToolModalProps } from '$lib/tools/tools';
	import { editImageRequest } from '../tools/editImage';

	let { context, onClose, initialTool }: MediaToolModalProps & { initialTool?: string } = $props();

	let request = $derived.by(() => {
		const base = editImageRequest(context);
		return base && initialTool ? { ...base, initialTool } : base;
	});

	async function handleResult(result: MediaEditorResult) {
		if (context.scope === 'library') {
			await applyLibraryPaintResult(
				result,
				context.items.map((item) => item.id)
			);
			return;
		}
		toasts.success('Saved to your library as a new image');
	}
</script>

<MediaEditors {request} {onClose} onResult={handleResult} />
