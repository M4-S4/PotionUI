<script lang="ts">
	import FieldShell from './FieldShell.svelte';
	import { ANCHOR_VALUES, anchorLabel, isAnchorValue } from './anchorValues';

	export let name: string | null;
	export let config: any = {};
	export let value: any;
	export let onChange: (fieldName: string, value: any) => void;

	$: label = config.title || name || '';
	$: description = config.description || '';
	$: tooltip = config.tooltip || config.configuration?.tooltip;
	$: disabled = config.disabled ?? false;
	$: current = isAnchorValue(value) ? value : 'center';

	let cells: HTMLButtonElement[] = [];

	function choose(anchor: string) {
		if (!disabled && name) onChange(name, anchor);
	}

	function move(event: KeyboardEvent, index: number) {
		const row = Math.floor(index / 3);
		const col = index % 3;
		const deltas: Record<string, [number, number]> = {
			ArrowUp: [-1, 0],
			ArrowDown: [1, 0],
			ArrowLeft: [0, -1],
			ArrowRight: [0, 1]
		};
		const delta = deltas[event.key];
		if (!delta) return;
		event.preventDefault();
		const nextRow = Math.min(2, Math.max(0, row + delta[0]));
		const nextCol = Math.min(2, Math.max(0, col + delta[1]));
		const next = nextRow * 3 + nextCol;
		choose(ANCHOR_VALUES[next]);
		cells[next]?.focus();
	}
</script>

<FieldShell {name} {label} {description} {tooltip} labelId={name ? `${name}-label` : undefined}>
	<div class="flex items-center gap-4">
		<div
			role="group"
			aria-label={label}
			class="inline-grid grid-cols-3 gap-1 rounded border border-line bg-surface-2 p-1"
		>
			{#each ANCHOR_VALUES as anchor, index (anchor)}
				<button
					type="button"
					bind:this={cells[index]}
					data-anchor={anchor}
					aria-label={anchorLabel(anchor)}
					aria-pressed={anchor === current}
					tabindex={anchor === current ? 0 : -1}
					{disabled}
					class="group flex h-8 w-8 items-center justify-center rounded border transition-colors disabled:cursor-not-allowed disabled:opacity-50 {anchor ===
					current
						? 'border-signal/60 bg-signal/10'
						: 'border-transparent hover:border-line-hover'}"
					on:click={() => choose(anchor)}
					on:keydown={(event) => move(event, index)}
				>
					<span
						class="block h-2 w-2 rounded-[2px] transition-colors {anchor === current
							? 'bg-signal'
							: 'bg-fg-disabled group-hover:bg-fg-muted'}"
					></span>
				</button>
			{/each}
		</div>
		<span class="font-mono text-sm text-fg-muted" data-anchor-readout>{anchorLabel(current)}</span>
	</div>
</FieldShell>
