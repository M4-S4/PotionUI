<script lang="ts">
	import { onMount } from 'svelte';
	import type { PresetModeVariant } from '$lib/types/api';
	import SessionScope from '$lib/components/session/SessionScope.svelte';
	import SessionControl from '$lib/components/session/SessionControl.svelte';

	export let presetId: string | null = null;
	export let currentMode: string | null = null;
	export let tabId: string;
	export let presetVersion: string | undefined = undefined;
	export let availableModes: Array<{ id: string; variants?: PresetModeVariant[] }> = [];

	let isClient = false;

	onMount(() => {
		isClient = true;
	});
</script>

<SessionScope {presetId} {currentMode} {tabId} {presetVersion} {availableModes} let:session let:actions let:fieldLabels>
	{#if isClient}
		<SessionControl {session} {actions} {fieldLabels} layout="sheet" />
	{/if}
</SessionScope>
