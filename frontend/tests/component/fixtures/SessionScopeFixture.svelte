<script lang="ts">
	import SessionScope from '$lib/components/session/SessionScope.svelte';
	import SessionControl from '$lib/components/session/SessionControl.svelte';

	export let presetId: string | null = null;
	export let currentMode: string | null = null;
	export let tabId: string;
	export let availableModes: Array<{ id: string }> = [];
	export let layout: 'segment' | 'sheet' = 'segment';
</script>

<SessionScope {presetId} {currentMode} {tabId} {availableModes} let:session let:actions let:fieldLabels>
	<SessionControl {session} {actions} {fieldLabels} {layout} />
</SessionScope>
