<script lang="ts">
	import type { Snippet } from 'svelte';

	let { titleExtra }: { titleExtra?: Snippet } = $props();
</script>

<header data-editor-title>
	{#if titleExtra}{@render titleExtra()}{/if}
</header>
