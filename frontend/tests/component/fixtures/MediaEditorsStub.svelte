<script lang="ts">
	import { getContext } from 'svelte';
	import { PAINT_INITIAL_TOOL } from '$lib/components/imageEditor/initialTool';

	let {
		request = null,
		onClose,
		onResult
	}: {
		request?: { kind: string; source: { url: string; fileName: string } } | null;
		onClose: () => void;
		onResult: (result: unknown, request: unknown) => void;
	} = $props();

	const initialTool = getContext<(() => string | null) | undefined>(PAINT_INITIAL_TOOL);
</script>

{#if request}
	<div
		data-editor-stub
		data-kind={request.kind}
		data-url={request.source.url}
		data-initial-tool={initialTool?.() ?? ''}
	>
		<button type="button" data-stub-close onclick={onClose}>close</button>
		<button
			type="button"
			data-stub-result
			onclick={() =>
				onResult(
					{
						type: 'item',
						item: { id: 'looked', filename: 'looked.png', media_type: 'image', url: '/looked.png' },
						replaced: false
					},
					request
				)}>result</button
		>
	</div>
{/if}
