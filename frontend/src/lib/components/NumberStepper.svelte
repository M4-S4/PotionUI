<script lang="ts">
	import Icon from './Icon.svelte';

	let {
		value,
		step = 1,
		decimals = 0,
		min = undefined,
		max = undefined,
		label,
		onChange
	}: {
		value: number;
		step?: number;
		decimals?: number;
		min?: number;
		max?: number;
		label: string;
		onChange: (value: number) => void;
	} = $props();

	let draft = $state<string | null>(null);

	function clamp(n: number): number {
		let next = n;
		if (min !== undefined) next = Math.max(next, min);
		if (max !== undefined) next = Math.min(next, max);
		return Number(next.toFixed(Math.min(Math.max(decimals, 0), 10)));
	}

	function nudge(direction: 1 | -1) {
		const unit = step > 0 ? step : 1;
		onChange(clamp(value + direction * unit));
	}

	function commitDraft() {
		if (draft === null) return;
		const raw = draft.trim().replace(',', '.');
		draft = null;
		const parsed = raw === '' ? NaN : Number(raw);
		if (!Number.isFinite(parsed)) return;
		onChange(clamp(parsed));
	}

	const shown = $derived(draft ?? value.toFixed(Math.min(Math.max(decimals, 0), 10)));
</script>

<div class="flex h-8 items-stretch rounded border border-line bg-surface-2 focus-within:border-signal max-md:h-11" role="group" aria-label={`${label} control`}>
	<button
		type="button"
		class="grid w-8 flex-shrink-0 place-items-center rounded-l text-fg-muted transition-colors hover:bg-surface-3 hover:text-fg disabled:cursor-not-allowed disabled:opacity-50 max-md:w-11"
		aria-label={`Decrease ${label}`}
		disabled={min !== undefined && value <= min}
		onclick={() => nudge(-1)}
	>
		<Icon name="minus" className="h-3.5 w-3.5" />
	</button>
	<input
		type="text"
		inputmode="decimal"
		class="min-w-0 flex-1 bg-transparent px-1 text-center font-mono text-sm tabular-nums text-fg outline-none"
		aria-label={label}
		value={shown}
		oninput={(e) => (draft = e.currentTarget.value)}
		onblur={commitDraft}
		onkeydown={(e) => {
			if (e.key === 'Enter') commitDraft();
			else if (e.key === 'ArrowUp') {
				e.preventDefault();
				nudge(1);
			} else if (e.key === 'ArrowDown') {
				e.preventDefault();
				nudge(-1);
			}
		}}
	/>
	<button
		type="button"
		class="grid w-8 flex-shrink-0 place-items-center rounded-r text-fg-muted transition-colors hover:bg-surface-3 hover:text-fg disabled:cursor-not-allowed disabled:opacity-50 max-md:w-11"
		aria-label={`Increase ${label}`}
		disabled={max !== undefined && value >= max}
		onclick={() => nudge(1)}
	>
		<Icon name="plus" className="h-3.5 w-3.5" />
	</button>
</div>
