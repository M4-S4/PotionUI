<script lang="ts">
	import Tooltip from '$lib/components/Tooltip.svelte';
	import IconButton from '$lib/components/ui/IconButton.svelte';
	import Button from '$lib/components/ui/Button.svelte';
</script>

<Tooltip text="Delete the shot" delay={0}>
	<IconButton icon="trash" label="Delete" />
</Tooltip>
<Tooltip text="Render every take" delay={0}>
	<Button>Render</Button>
</Tooltip>
<Tooltip text="Plain child" delay={0}>
	<input class="plain-child" aria-label="Plain" />
</Tooltip>
