<script lang="ts">
	import { onMount } from 'svelte';

	let { onProbeMount }: { onProbeMount: () => void } = $props();

	onMount(() => {
		onProbeMount();
	});
</script>

<div data-mount-probe></div>
