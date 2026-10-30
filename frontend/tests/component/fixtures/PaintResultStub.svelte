<script lang="ts">
	let {
		request = null,
		onClose,
		onResult
	}: {
		request?: { kind: string } | null;
		onClose: () => void;
		onResult: (result: unknown, request: unknown) => void | Promise<void>;
	} = $props();

	const created = {
		id: 'row-new',
		filename: 'new-uuid.png',
		original_filename: 'portrait-edit.png',
		media_type: 'image',
		url: '/api/media/uploads/new-uuid.png',
		width: 800,
		height: 600
	};
	const replacement = {
		id: 'row-1',
		filename: 'replaced-uuid.png',
		original_filename: 'portrait.png',
		media_type: 'image',
		url: '/api/media/uploads/replaced-uuid.png',
		width: 800,
		height: 600
	};
</script>

{#if request}
	<div data-paint-stub>
		<button type="button" data-paint-close onclick={onClose}>close</button>
		<button
			type="button"
			data-paint-new
			onclick={() => onResult({ type: 'item', item: created, replaced: false }, request)}>new</button
		>
		<button
			type="button"
			data-paint-replace
			onclick={() => onResult({ type: 'item', item: replacement, replaced: true }, request)}>replace</button
		>
	</div>
{/if}
