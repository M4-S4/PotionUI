<script lang="ts">
	import StudioSheet from '$lib/components/StudioSheet.svelte';
	import UserPanel from './UserPanel.svelte';

	export let onClose: () => void;
</script>

<StudioSheet ariaLabel="Accounts" maxHeight="92%" on:close={onClose}>
	<UserPanel variant="sheet" {onClose} />
</StudioSheet>
