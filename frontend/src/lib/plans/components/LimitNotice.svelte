<script lang="ts">
	import { refusalDismissed, serverRefusal, submitGate } from '../store';
	import LimitNoticeCard from './LimitNoticeCard.svelte';

	let visible = $derived(!!$serverRefusal && !$refusalDismissed && $submitGate !== null);
</script>

{#if visible}
	<LimitNoticeCard />
{/if}
