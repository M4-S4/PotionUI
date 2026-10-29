<script lang="ts">
	import { IconButton } from '$lib/components/ui';

	let {
		label,
		progress,
		cancelling = false,
		onCancel,
		bottom = 0
	}: {
		label: string;
		progress: number;
		cancelling?: boolean;
		onCancel: () => void;
		bottom?: number;
	} = $props();

	let percent = $derived(Math.round(Math.min(1, Math.max(0, progress)) * 100));
</script>

<div
	class="run-band absolute inset-x-0 z-40 grid grid-cols-[1fr_auto] items-center gap-x-2 gap-y-1 bg-scrim/75 p-2 text-scrim-on backdrop-blur-sm"
	style="bottom: {bottom}px"
	data-tool-run
>
	<span
		class="run-label min-w-0 truncate font-mono text-[12px] leading-4 uppercase tracking-[0.07em] text-scrim-on/70"
		data-tool-run-label>{label}</span
	>
	<span class="flex items-center gap-1.5 justify-self-end">
		<span
			class="font-mono text-[12px] leading-4 tabular-nums"
			role="progressbar"
			aria-label="{label} progress"
			aria-valuemin={0}
			aria-valuemax={100}
			aria-valuenow={percent}
			data-tool-run-percent>{percent}%</span
		>
		<IconButton
			icon="close"
			label="Cancel {label}"
			size="xs"
			disabled={cancelling}
			class="-my-1 -mr-1 !bg-transparent !text-scrim-on/80 hover:!bg-scrim-on/15 hover:!text-scrim-on"
			onclick={(event: MouseEvent) => {
				event.stopPropagation();
				onCancel();
			}}
		/>
	</span>
	<div class="relative col-span-2 h-1 overflow-hidden rounded-sm bg-scrim-on/20">
		<div
			class="absolute inset-y-0 left-0 bg-signal transition-[width] duration-200 ease-linear"
			style="width: {percent}%"
			data-tool-run-fill
		></div>
		<div
			class="pointer-events-none absolute inset-0"
			style="background: repeating-linear-gradient(90deg, transparent 0 9px, rgb(var(--scrim) / 0.45) 9px 10px)"
		></div>
	</div>
</div>

<style>
	.run-band {
		container-type: inline-size;
	}
	@container (max-width: 200px) {
		.run-label {
			display: none;
		}
	}
</style>
