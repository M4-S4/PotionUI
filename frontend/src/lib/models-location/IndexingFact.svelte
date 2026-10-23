<script lang="ts">
	let {
		label,
		value,
		wide = false
	}: {
		label: string;
		value: string;
		wide?: boolean;
	} = $props();
</script>

<div class="min-w-0 rounded border border-line bg-surface-2 px-3 py-2 {wide ? 'max-sm:col-span-2' : ''}">
	<div class="text-sm text-fg-subtle">{label}</div>
	<div class="mt-0.5 truncate font-mono text-md font-medium tabular-nums text-fg">{value}</div>
</div>
