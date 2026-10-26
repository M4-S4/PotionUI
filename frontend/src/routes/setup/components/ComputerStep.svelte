<script lang="ts">
	import type { SetupMachine, SetupRecipe } from '$lib/services/api/setup';
	import { Button, Spinner } from '$lib/components/ui';
	import StepHeader from '$lib/components/setup/StepHeader.svelte';
	import StepFacts from '$lib/components/setup/StepFacts.svelte';
	import MachineSummary from '$lib/components/setup/MachineSummary.svelte';
	import { stepStatus } from '$lib/utils/setupGuide';

	let {
		machine,
		loading,
		recipes,
		onContinue
	}: {
		machine: SetupMachine | null;
		loading: boolean;
		recipes: SetupRecipe[];
		onContinue: () => void;
	} = $props();

	const FACTS = [
		{
			label: 'Why we check',
			body: [
				'Bigger models need more graphics memory. Suggesting one that does not fit would end in a failed first image.'
			]
		},
		{
			label: 'What it reads',
			body: ['Graphics card name and memory, system memory, and free space on the drive where models will live.']
		},
		{
			label: 'Not detected right?',
			body: ['Continue anyway. Every model starts with a fit note you can ignore.']
		}
	];
</script>

<section class="flex flex-col gap-5" aria-labelledby="h-computer">
	<StepHeader
		step="computer"
		titleId="h-computer"
		title="Your computer"
		lede="This is what PotionUI found. We use it to suggest a starter model that fits. Nothing is changed and nothing is sent anywhere."
		status={stepStatus('computer')}
	/>

	{#if loading}
		<div class="flex justify-center py-10"><Spinner size="md" /></div>
	{:else}
		<MachineSummary {machine} {recipes} />
	{/if}

	<StepFacts facts={FACTS} />

	<div class="flex flex-wrap items-center gap-3">
		<Button variant="primary" size="lg" class="max-sm:w-full" onclick={onContinue}>Continue</Button>
	</div>
</section>
