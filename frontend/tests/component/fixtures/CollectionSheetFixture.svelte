<script lang="ts">
	import CollectionSidebarSheet from '$lib/components/collections/CollectionSidebarSheet.svelte';
	import CollectionLibrarySidebar from '$lib/components/collections/CollectionLibrarySidebar.svelte';

	let {
		sidebar,
		collapsed = false,
		onExpand
	}: {
		sidebar: any;
		collapsed?: boolean;
		onExpand?: () => void;
	} = $props();
</script>

<CollectionSidebarSheet label="History" {collapsed} {onExpand}>
	<CollectionLibrarySidebar {...sidebar} />
</CollectionSidebarSheet>
