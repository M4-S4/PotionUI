<script lang="ts">
	import type { Snippet } from 'svelte';

	let { titleExtra, placeholder }: { titleExtra?: Snippet; placeholder?: string } = $props();
</script>

<header data-editor-title>
	{#if titleExtra}{@render titleExtra()}{/if}
</header>
<div data-editor-placeholder>{placeholder}</div>
