<script lang="ts">
	import Tooltip from '$lib/components/Tooltip.svelte';
	import IconButton from '$lib/components/ui/IconButton.svelte';
</script>

<Tooltip text="Close" delay={0}>
	<IconButton icon="close" label="Close" />
</Tooltip>
