<script lang="ts">
	import Layout from '../../../src/routes/+layout.svelte';
	import MountProbe from './MountProbe.svelte';

	let { onProbeMount }: { onProbeMount: () => void } = $props();
</script>

<Layout>
	<MountProbe {onProbeMount} />
</Layout>
