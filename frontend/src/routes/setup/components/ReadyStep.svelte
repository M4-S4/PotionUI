<script lang="ts">
	import { Button, Card } from '$lib/components/ui';
	import StepHeader from '$lib/components/setup/StepHeader.svelte';
	import StepFacts from '$lib/components/setup/StepFacts.svelte';

	let {
		installedName,
		thumbnailUrl,
		modelsPath,
		onCreate
	}: {
		installedName: string | null;
		thumbnailUrl: string | null;
		modelsPath: string;
		onCreate: () => void;
	} = $props();

	const NEXT_STEPS = [
		{
			title: 'Connect ComfyUI',
			body: 'Already run ComfyUI? Add it as an extra engine. PotionUI works without it.',
			label: 'Admin → Backends',
			href: '/admin?tab=backends'
		},
		{
			title: 'Add another models folder',
			body: 'Reuse models from Forge, A1111 or any other folder.',
			label: 'Admin → Models → Folders',
			href: '/admin?tab=models&view=folders'
		},
		{
			title: 'Invite people',
			body: 'Sign-up is closed by default. Add accounts for family or teammates yourself.',
			label: 'Admin → Users',
			href: '/admin?tab=users'
		},
		{
			title: 'Install more starters',
			body: 'Video, music, 3D and more image models.',
			label: 'Admin → Recipes',
			href: '/admin?tab=recipes'
		}
	];

	const lede = $derived(
		installedName
			? `${installedName} is installed and a test image was made on this computer. Open Generate to try your own prompt.`
			: 'Your account is ready. Install a starter model any time from Setup, then generate.'
	);
	const facts = $derived([
		{ label: 'Your account lives in', body: [{ mono: 'storage/db.sqlite in your PotionUI folder' }] },
		{ label: 'Your models live in', body: [{ mono: modelsPath }] },
		{ label: 'Look again any time', body: ['This checklist stays at Setup in the sidebar.'] }
	]);
</script>

<section class="flex flex-col gap-5" aria-labelledby="h-ready">
	<Card class="border-success/45 bg-success/5">
		<div class="grid grid-cols-[auto_1fr] items-center gap-5 max-sm:grid-cols-1">
			{#if thumbnailUrl}
				<img
					src={thumbnailUrl}
					alt="Output from your setup's test generation"
					class="aspect-square w-full max-w-[200px] rounded-lg border border-line-strong bg-black object-cover"
				/>
			{:else}
				<div
					class="grid aspect-square w-full max-w-[200px] place-items-center rounded-lg border border-line-strong bg-black"
					aria-hidden="true"
				>
					<span class="px-6 text-center font-mono text-sm uppercase tracking-[0.06em] text-white/60">
						First image
					</span>
				</div>
			{/if}
			<div class="flex flex-col gap-3">
				<StepHeader
					step="ready"
					titleId="h-ready"
					title="You're all set"
					lede={lede}
					status={{ text: 'Done', tone: 'success' }}
				/>
				<div>
					<Button variant="primary" size="lg" class="max-sm:w-full" onclick={onCreate}>
						Create your first image
					</Button>
				</div>
			</div>
		</div>
	</Card>

	<div>
		<p class="mb-2 font-mono text-sm uppercase tracking-[0.06em] text-fg-subtle">Optional next steps</p>
		<Card padding="none" class="overflow-hidden">
			{#each NEXT_STEPS as item, i (item.title)}
				<div
					class="grid grid-cols-[1fr_auto] items-center gap-3 px-4 py-3 max-sm:grid-cols-1 {i > 0
						? 'border-t border-line'
						: ''}"
				>
					<div>
						<div class="font-medium text-fg">{item.title}</div>
						<div class="text-fg-muted">{item.body}</div>
					</div>
					<a class="font-medium text-signal hover:underline" href={item.href}>{item.label}</a>
				</div>
			{/each}
		</Card>
	</div>

	<StepFacts {facts} />
</section>
