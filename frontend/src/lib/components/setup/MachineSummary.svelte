<script lang="ts">
	import type { SetupMachine, SetupRecipe } from '$lib/services/api/setup';
	import Card from '$lib/components/ui/Card.svelte';
	import Badge from '$lib/components/ui/Badge.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import { formatBytes } from '$lib/utils/format';
	import { diskStatus, machineVerdict, memoryStatus, ramGb, vramGb } from '$lib/utils/setupGuide';

	let {
		machine,
		recipes = []
	}: {
		machine: SetupMachine | null;
		recipes?: SetupRecipe[];
	} = $props();

	const verdict = $derived(machineVerdict(machine));
	const vram = $derived(vramGb(machine));
	const ram = $derived(ramGb(machine));
	const memory = $derived(ram === null ? null : memoryStatus(ram));
	const disk = $derived(machine?.models_disk ? diskStatus(machine.models_disk.free_bytes, recipes) : null);
	const warn = $derived(verdict.kind === 'no-gpu' || verdict.kind === 'low-memory');
</script>

<div class="space-y-5">
	<Card padding="none" class="overflow-hidden" >
		<div class="grid grid-cols-[28px_1fr_auto] items-center gap-3 px-4 py-3 max-sm:grid-cols-[24px_1fr]" data-row="gpu">
			<Icon name="cpu" className="h-5 w-5 text-fg-muted" />
			<div class="min-w-0">
				<p class="text-fg-subtle">Graphics card</p>
				{#if machine?.gpu}
					<p class="break-words font-medium text-fg">
						{machine.gpu.name}
						{#if vram !== null}
							<span class="font-mono font-normal tabular-nums text-fg-muted">· {vram} GB</span>
						{/if}
					</p>
				{:else if machine}
					<p class="font-medium text-fg">No NVIDIA graphics card found</p>
				{:else}
					<p class="font-medium text-fg-muted">Couldn't detect</p>
				{/if}
			</div>
			{#if machine?.gpu}
				<Badge variant={machine.gpu.supported ? 'success' : 'warning'} class="justify-self-end font-mono uppercase tracking-[0.04em] max-sm:col-start-2 max-sm:justify-self-start">
					{machine.gpu.supported ? 'Supported' : 'Not supported'}
				</Badge>
			{:else if machine}
				<Badge variant="warning" class="justify-self-end font-mono uppercase tracking-[0.04em] max-sm:col-start-2 max-sm:justify-self-start">Not supported</Badge>
			{/if}
		</div>

		<div class="grid grid-cols-[28px_1fr_auto] items-center gap-3 border-t border-line px-4 py-3 max-sm:grid-cols-[24px_1fr]" data-row="ram">
			<Icon name="server" className="h-5 w-5 text-fg-muted" />
			<div class="min-w-0">
				<p class="text-fg-subtle">System memory</p>
				{#if ram !== null}
					<p class="font-mono font-medium tabular-nums text-fg">{ram} GB</p>
				{:else}
					<p class="font-medium text-fg-muted">Couldn't detect</p>
				{/if}
			</div>
			{#if memory}
				<Badge variant={memory.tone} class="justify-self-end font-mono uppercase tracking-[0.04em] max-sm:col-start-2 max-sm:justify-self-start">{memory.text}</Badge>
			{/if}
		</div>

		<div class="grid grid-cols-[28px_1fr_auto] items-center gap-3 border-t border-line px-4 py-3 max-sm:grid-cols-[24px_1fr]" data-row="disk">
			<Icon name="database" className="h-5 w-5 text-fg-muted" />
			<div class="min-w-0">
				<p class="text-fg-subtle">Free disk space</p>
				{#if machine?.models_disk}
					<p class="font-medium text-fg">
						<span class="font-mono tabular-nums">{formatBytes(machine.models_disk.free_bytes, 0)}</span>
						<span class="font-normal text-fg-muted">on the drive that will hold models</span>
					</p>
				{:else}
					<p class="font-medium text-fg-muted">Couldn't detect</p>
				{/if}
			</div>
			{#if disk}
				<Badge variant={disk.tone} class="justify-self-end font-mono uppercase tracking-[0.04em] max-sm:col-start-2 max-sm:justify-self-start">{disk.text}</Badge>
			{/if}
		</div>
	</Card>

	<div
		class="grid grid-cols-[20px_1fr] gap-3 rounded-lg border px-3.5 py-3 {warn
			? 'border-warning/40 bg-warning/5'
			: 'border-line-strong bg-surface-2'}"
		role="note"
		data-verdict={verdict.kind}
	>
		<Icon name={warn ? 'warning' : 'check-circle'} className="mt-px h-5 w-5 {warn ? 'text-warning' : 'text-fg-muted'}" />
		<p class="text-fg-muted"><strong class="font-semibold text-fg">{verdict.title}</strong> {verdict.body}</p>
	</div>
</div>
