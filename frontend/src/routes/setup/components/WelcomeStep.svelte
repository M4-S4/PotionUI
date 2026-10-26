<script lang="ts">
	import { Badge, Button, Card } from '$lib/components/ui';
	import Icon from '$lib/components/Icon.svelte';
	import StepHeader from '$lib/components/setup/StepHeader.svelte';
	import StepFacts from '$lib/components/setup/StepFacts.svelte';

	let { onStart }: { onStart: () => void } = $props();

	const TRUST = [
		{
			icon: 'monitor',
			title: 'Your account and images stay here',
			body: 'Your account, settings and generated images are saved on this computer. PotionUI has no cloud account to sign up for.'
		},
		{
			icon: 'download',
			title: 'The internet is used to download models',
			body: 'Setup downloads model files from Hugging Face, and only after you approve the exact list on the install step. Cloud services are used only if you connect one.'
		},
		{
			icon: 'plug',
			title: 'ComfyUI is optional',
			body: 'PotionUI generates by itself. If you already use ComfyUI you can connect it later as an extra engine. You do not need it.'
		}
	];

	const PLAN = [
		{
			title: 'Create your owner account',
			body: 'Stored on this computer, used to sign in here.',
			tag: 'Required'
		},
		{ title: 'Check your computer', body: 'Graphics card, memory and free disk space.', tag: '' },
		{
			title: 'Choose where models are kept',
			body: "PotionUI's own folder, or one you already have.",
			tag: 'Can change later'
		},
		{
			title: 'Pick and install a starter model',
			body: 'You see the files, sizes and source before anything downloads.',
			tag: 'Optional'
		}
	];

	const FACTS = [
		{
			label: 'You will need',
			body: ['An NVIDIA graphics card with 8 GB of memory or more (GTX 16-series, RTX 20-series or newer).']
		},
		{
			label: 'Disk space',
			body: [
				'A few GB for the smallest image starter, tens of GB for the largest. You see the exact size before anything downloads.'
			]
		},
		{ label: 'Internet', body: ['Only for the model download that setup asks you to approve.'] }
	];
</script>

<section class="flex flex-col gap-5" aria-labelledby="h-welcome">
	<StepHeader
		step="welcome"
		titleId="h-welcome"
		title="A local studio for AI images and video"
		lede="PotionUI runs on this computer and uses its graphics card to generate. Setup creates your account, checks your machine, and installs one starter model so you can make a first image."
	/>

	<Card padding="none" class="overflow-hidden">
		<div class="border-b border-line px-4 py-3 font-mono text-sm uppercase tracking-[0.06em] text-fg">
			Runs on your computer
		</div>
		{#each TRUST as item, i (item.title)}
			<div class="grid grid-cols-[28px_1fr] gap-3 px-4 py-3.5 {i > 0 ? 'border-t border-line' : ''}">
				<Icon name={item.icon} className="mt-0.5 h-5 w-5 text-fg-muted" />
				<div>
					<h3 class="font-semibold text-fg">{item.title}</h3>
					<p class="text-fg-muted">{item.body}</p>
				</div>
			</div>
		{/each}
	</Card>

	<Card>
		<div class="mb-3 flex items-start justify-between gap-3">
			<h2 class="text-lg font-semibold text-fg">What setup will do</h2>
			<Badge class="font-mono uppercase tracking-[0.04em]">Sizes shown before download</Badge>
		</div>
		<ol class="m-0 list-none p-0">
			{#each PLAN as item, i (item.title)}
				<li
					class="grid grid-cols-[28px_1fr_auto] items-baseline gap-3 py-2.5 first:pt-0 last:pb-0 max-sm:grid-cols-[28px_1fr] {i > 0
						? 'border-t border-line'
						: ''}"
				>
					<span
						class="grid h-[22px] w-[22px] place-items-center self-start rounded border border-line-strong font-mono text-sm text-fg-subtle"
					>
						{i + 1}
					</span>
					<div>
						<div class="font-medium text-fg">{item.title}</div>
						<div class="text-fg-muted">{item.body}</div>
					</div>
					{#if item.tag}
						<Badge class="whitespace-nowrap font-mono uppercase tracking-[0.04em] max-sm:col-start-2 max-sm:w-fit">
							{item.tag}
						</Badge>
					{/if}
				</li>
			{/each}
		</ol>
	</Card>

	<StepFacts facts={FACTS} />

	<div class="flex items-center gap-3">
		<Button variant="primary" size="lg" class="max-sm:w-full" onclick={onStart}>Get started</Button>
	</div>
</section>
