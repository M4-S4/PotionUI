<script lang="ts">
	import type { SetupMachine, SetupRecipe } from '$lib/services/api/setup';
	import { Button, EmptyState, Spinner } from '$lib/components/ui';
	import Icon from '$lib/components/Icon.svelte';
	import StepHeader from '$lib/components/setup/StepHeader.svelte';
	import StepFacts from '$lib/components/setup/StepFacts.svelte';
	import StarterCard from '$lib/components/setup/StarterCard.svelte';
	import {
		orderStarters,
		recommendStarter,
		reviewDownloadLabel,
		starterSourceText,
		stepStatus
	} from '$lib/utils/setupGuide';

	let {
		recipes,
		loading,
		error,
		machine,
		modelsPath,
		starting,
		startError,
		onStart,
		onSkip,
		onBack,
		onRetry
	}: {
		recipes: SetupRecipe[] | null;
		loading: boolean;
		error: string;
		machine: SetupMachine | null;
		modelsPath: string;
		starting: boolean;
		startError: string;
		onStart: (recipe: SetupRecipe) => void;
		onSkip: () => void;
		onBack?: () => void;
		onRetry: () => void;
	} = $props();

	let picked = $state<string | null>(null);

	const recommendedId = $derived(recommendStarter(recipes ?? [], machine));
	const ordered = $derived(orderStarters(recipes ?? [], recommendedId));
	const selected = $derived(
		ordered.find((r) => r.id === picked) ?? ordered.find((r) => r.id === recommendedId) ?? null
	);
	const noGpu = $derived(machine !== null && (!machine.gpu || machine.gpu.supported === false));
	const sources = $derived(
		Array.from(new Set(ordered.flatMap((r) => r.sources))).join(', ') ||
			(selected ? starterSourceText(selected) : 'Not stated')
	);
	const facts = $derived([
		{
			label: 'What happens next',
			body: ['You see every file, its size and where it comes from. Nothing downloads until you approve.']
		},
		{ label: 'Download from', body: [{ mono: sources }] },
		{ label: 'Saved to', body: [{ mono: modelsPath }] }
	]);
</script>

<section class="flex flex-col gap-5" aria-labelledby="h-starter">
	<StepHeader
		step="starter"
		titleId="h-starter"
		title="Pick a starter model"
		lede="One model is enough to begin. Each starter downloads the files shown, then makes a test image so you know it works. More are in Admin → Recipes later."
		status={stepStatus('starter')}
	/>

	{#if noGpu}
		<div class="grid grid-cols-[20px_1fr] gap-3 rounded-lg border border-warning/40 bg-warning/5 px-3.5 py-3" role="note">
			<Icon name="warning" className="mt-px h-5 w-5 text-warning" />
			<p class="text-fg-muted">
				<strong class="font-semibold text-fg">No NVIDIA graphics card was found.</strong> These starters
				run on your own GPU, so installing one here would not work. Skip for now and connect a cloud
				provider later, or go back if the detection is wrong.
			</p>
		</div>
	{/if}

	{#if loading || (recipes === null && !error)}
		<div class="flex justify-center py-10"><Spinner size="md" /></div>
	{:else if error}
		<EmptyState title="Couldn't load starter models" description={error} icon="warning">
			{#snippet actions()}
				<Button variant="secondary" onclick={onRetry}>Try again</Button>
			{/snippet}
		</EmptyState>
	{:else if ordered.length === 0}
		<EmptyState
			title="No starter models available"
			description="There's nothing to install from here right now. You can add models later in Admin → Recipes."
			icon="box"
		/>
	{:else}
		<div class="flex flex-col gap-3" role="radiogroup" aria-label="Starter models">
			{#each ordered as recipe, i (recipe.id)}
				<StarterCard
					{recipe}
					{machine}
					checked={selected?.id === recipe.id}
					tabbable={selected ? selected.id === recipe.id : i === 0}
					recommended={recipe.id === recommendedId}
					onselect={() => (picked = recipe.id)}
				/>
			{/each}
		</div>

		<details class="group">
			<summary class="cursor-pointer list-none rounded-sm font-medium text-signal hover:underline">
				More starters: video, music, 3D, image editing and other models
			</summary>
			<p class="mt-2.5 text-fg-subtle">
				These appear in Admin → Recipes after setup. They are larger or need a stronger card, so we
				keep them out of your first run.
			</p>
		</details>

		<StepFacts {facts} />

		{#if startError}
			<p class="text-base text-danger" role="alert">{startError}</p>
		{/if}

		<div class="flex flex-wrap items-center gap-3 max-sm:[&>*:first-child]:w-full">
			<Button
				variant="primary"
				size="lg"
				loading={starting}
				disabled={!selected || noGpu}
				onclick={() => selected && onStart(selected)}
			>
				{reviewDownloadLabel(selected)}
			</Button>
			<Button variant="secondary" onclick={onSkip}>Skip for now</Button>
			{#if onBack}
				<Button variant="ghost" onclick={onBack}>Back</Button>
			{/if}
		</div>
	{/if}

	{#if !loading && (error || ordered.length === 0)}
		<div class="flex flex-wrap items-center gap-3">
			<Button variant="secondary" onclick={onSkip}>Skip for now</Button>
			{#if onBack}
				<Button variant="ghost" onclick={onBack}>Back</Button>
			{/if}
		</div>
	{/if}
</section>
