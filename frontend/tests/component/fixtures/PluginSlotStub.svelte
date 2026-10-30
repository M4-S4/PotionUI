<script lang="ts">
	let {
		hookName,
		context = {},
		excludePlugins = []
	}: { hookName: string; context?: Record<string, unknown>; excludePlugins?: string[] } = $props();
</script>

<div
	class="plugin-slot"
	data-plugin-slot-stub
	data-hook={hookName}
	data-context={JSON.stringify(context)}
	data-exclude={excludePlugins.join(',')}
></div>
