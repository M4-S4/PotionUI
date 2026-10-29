<script lang="ts">
	import { onMount } from 'svelte';
	import MediaToolDialog from '$lib/media-tools/MediaToolDialog.svelte';
	import { loadMediaToolCatalog, mediaToolCatalog, toolById } from '$lib/media-tools/catalog';
	import type { MediaToolItem, MediaToolSource } from '$lib/media-tools/types';
	import {
		toolIdOfKind,
		type MediaEditorRequest,
		type MediaEditorResult,
		type ToolDialogKind
	} from './types';

	let {
		request,
		detach = false,
		onClose,
		onResult
	}: {
		request: MediaEditorRequest;
		detach?: boolean;
		onClose: () => void;
		onResult: (result: MediaEditorResult, request: MediaEditorRequest) => void | Promise<void>;
	} = $props();

	let tool = $derived(toolById(toolIdOfKind(request.kind as ToolDialogKind), $mediaToolCatalog));

	let source = $derived<MediaToolSource>({
		id:
			request.source.key ??
			request.source.itemId ??
			request.source.generationFileId ??
			request.source.storedPath ??
			request.source.url,
		kind: request.source.kind,
		url: request.source.url,
		filename: request.source.fileName,
		itemId: request.source.itemId ?? null,
		generationFileId: request.source.generationFileId ?? null,
		storedPath: request.source.storedPath ?? null,
		width: request.source.width ?? null,
		height: request.source.height ?? null,
		durationSeconds: request.source.durationSeconds ?? null,
		fps: request.source.fps ?? null
	});

	onMount(() => {
		void loadMediaToolCatalog();
	});

	function handleResult(result: { items: MediaToolItem[]; replaced: boolean }) {
		const mapped: MediaEditorResult =
			result.items.length === 1
				? { type: 'item', item: result.items[0], replaced: result.replaced }
				: { type: 'items', items: result.items };
		return onResult(mapped, request);
	}
</script>

{#if tool}
	<MediaToolDialog {tool} items={[source]} scope="field" {detach} {onClose} onResult={handleResult} />
{/if}
