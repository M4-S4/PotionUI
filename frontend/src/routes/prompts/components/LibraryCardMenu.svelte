<script lang="ts">
	import Menu, { type MenuItem } from '$lib/components/ui/Menu.svelte';

	let {
		items,
		onselect,
		open = $bindable(false)
	}: {
		items: MenuItem[];
		onselect: (id: string) => void;
		open?: boolean;
	} = $props();

	function stop(event: Event) {
		event.stopPropagation();
	}
</script>

<span class="inline-flex" role="presentation" onclick={stop} onkeydown={stop}>
	<Menu label="More actions" tooltip="More actions" {items} {onselect} bind:open />
</span>
