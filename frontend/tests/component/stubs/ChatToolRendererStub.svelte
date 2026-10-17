<script lang="ts">
	let { execution, expanded }: { execution: { tool_name: string }; expanded?: boolean } = $props();
</script>

<div data-testid="plugin-renderer" data-expanded={expanded ? 'true' : 'false'}>{execution.tool_name}</div>
