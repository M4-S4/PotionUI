<script lang="ts">
	let {
		context,
		onClose,
		onDone,
		lookId = undefined
	}: {
		context: { items: unknown[] };
		onClose: () => void;
		onDone: () => void;
		lookId?: string;
	} = $props();
</script>

<div data-core-stub data-look-id={lookId ?? ''} data-item-count={context.items.length}>
	<button type="button" data-stub-close onclick={onClose}>close</button>
	<button type="button" data-stub-done onclick={onDone}>done</button>
</div>
