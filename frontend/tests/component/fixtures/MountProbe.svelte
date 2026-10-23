<script lang="ts">
	let { label = 'start' }: { label?: string } = $props();
</script>

<p data-testid="probe">{label}</p>
