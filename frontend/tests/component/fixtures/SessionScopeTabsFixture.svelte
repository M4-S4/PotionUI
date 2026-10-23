<script lang="ts">
	import SessionScope from '$lib/components/session/SessionScope.svelte';

	export let tabIds: string[] = [];
	export let activeTabId = '';
	export let presetId: string | null = null;
	export let currentMode: string | null = null;
</script>

{#each tabIds as id (id)}
	{#if id === activeTabId}
		<SessionScope {presetId} {currentMode} tabId={id} />
	{/if}
{/each}
