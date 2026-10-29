<script lang="ts">
	let {
		onClose,
		onResult,
		onDone,
		...rest
	}: {
		onClose?: () => void;
		onResult?: (...args: unknown[]) => unknown;
		onDone?: () => void;
		[key: string]: unknown;
	} = $props();

	const item = { id: 'out-1', filename: 'out.png', media_type: 'image', url: '/out.png' };
	let summary = $derived(
		JSON.stringify(rest, (_key, value) => (value instanceof Set ? [...value] : typeof value === 'function' ? 'fn' : value))
	);
</script>

<div data-host-stub data-props={summary}>
	<button type="button" data-stub-close onclick={() => onClose?.()}>close</button>
	<button
		type="button"
		data-stub-result-editor
		onclick={() => onResult?.({ type: 'item', item, replaced: true }, rest.request)}>editor result</button
	>
	<button
		type="button"
		data-stub-result-editor-items
		onclick={() => onResult?.({ type: 'items', items: [item, item] }, rest.request)}>editor items</button
	>
	<button
		type="button"
		data-stub-result-mask
		onclick={() => onResult?.({ type: 'mask', maskPath: 'x.png' }, rest.request)}>editor mask</button
	>
	<button
		type="button"
		data-stub-result-dialog
		onclick={() => onResult?.({ items: [item], replaced: false })}>dialog result</button
	>
	<button type="button" data-stub-done onclick={() => onDone?.()}>done</button>
</div>
