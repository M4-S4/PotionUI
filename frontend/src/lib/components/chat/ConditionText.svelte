<script lang="ts">
	import { readable } from 'svelte/store';
	import ConditionChip from '$lib/components/ConditionChip.svelte';
	import { splitConditionText } from '$lib/chat/conditionText';
	import type { VariableRoll, VariablesMap } from '$lib/utils/variableDefs';

	let {
		text,
		variables = undefined,
		rolls = undefined
	}: {
		text: string;
		variables?: VariablesMap;
		rolls?: Record<string, VariableRoll>;
	} = $props();

	const parts = $derived(splitConditionText(text));
	const context = $derived(readable({ variables: variables ?? {}, rolls: rolls ?? {} }));
</script>

{#each parts as part, i (i)}{#if part.kind === 'condition'}{#key part.raw}<ConditionChip
				raw={part.raw}
				{context}
				readonly
				checkVariables={variables !== undefined}
			/>{/key}{:else}{part.text}{/if}{/each}
