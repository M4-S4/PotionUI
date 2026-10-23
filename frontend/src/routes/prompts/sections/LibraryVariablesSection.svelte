<script lang="ts">
	import VariableManagerModal from '$lib/components/VariableManagerModal.svelte';
	import VariablesListSection from '$lib/components/VariablesListSection.svelte';
	import type { LibraryTabVariables, LibraryVariableScope } from '$lib/prompts/libraryTabVariables';

	let {
		source,
		scope,
		open = $bindable(false)
	}: {
		source: LibraryTabVariables;
		scope: LibraryVariableScope;
		open?: boolean;
	} = $props();
</script>

<VariablesListSection
	variables={source.variables}
	emptyText="The active Generate tab has no variables."
	note="Conditions and variable chips resolve against the active Generate tab."
	onManage={() => (open = true)}
/>

<VariableManagerModal
	isOpen={open}
	variables={source.variables}
	usageCounts={scope.usageCounts}
	readers={scope.readers}
	on:close={() => (open = false)}
	on:change={(event) => source.setAll(event.detail)}
/>
