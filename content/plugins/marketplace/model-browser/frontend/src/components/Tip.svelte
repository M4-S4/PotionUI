<script>
	import HostUi from '../../../../../sdk/HostUi.svelte';

	let { text = '', position = 'top', wrapperClass = undefined, children } = $props();
</script>

{#if text}
	<HostUi name="Tooltip" props={{ text, position, ...(wrapperClass ? { wrapperClass } : {}) }}>
		{@render children?.()}
	</HostUi>
{:else}
	{@render children?.()}
{/if}
