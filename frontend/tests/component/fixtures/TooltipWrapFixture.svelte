<script lang="ts">
	import Tooltip from '$lib/components/Tooltip.svelte';
	export let text: string;
	export let truncateOnly = false;
</script>

<Tooltip {text} {truncateOnly} delay={0}>
	<button type="button">trigger</button>
</Tooltip>
