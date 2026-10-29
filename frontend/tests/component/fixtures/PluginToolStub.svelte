<script lang="ts">
	let {
		generationIds = [],
		generations = [],
		items = undefined,
		onClose,
		onDone
	}: {
		generationIds?: string[];
		generations?: unknown[];
		items?: unknown[];
		onClose: () => void;
		onDone: () => void;
	} = $props();
</script>

<div
	data-plugin-stub
	data-items={JSON.stringify(items ?? null)}
	data-generation-ids={JSON.stringify(generationIds)}
	data-generation-count={generations.length}
>
	<button type="button" data-stub-done onclick={onDone}>done</button>
	<button type="button" data-stub-close onclick={onClose}>close</button>
</div>
