<script lang="ts">
	import type { Snippet } from 'svelte';
	import Badge from './Badge.svelte';

	let {
		checked,
		tabbable = checked,
		title,
		badge,
		description,
		class: className = '',
		onselect,
		children
	}: {
		checked: boolean;
		tabbable?: boolean;
		title: string;
		badge?: string;
		description?: string;
		class?: string;
		onselect: () => void;
		children?: Snippet;
	} = $props();

	const ARROW_STEP: Record<string, number> = {
		ArrowDown: 1,
		ArrowRight: 1,
		ArrowUp: -1,
		ArrowLeft: -1
	};

	function handleKeydown(event: KeyboardEvent) {
		const step = ARROW_STEP[event.key];
		if (step === undefined) return;
		event.preventDefault();
		const current = event.currentTarget as HTMLElement;
		const group = current.closest('[role="radiogroup"]');
		if (!group) return;
		const radios = Array.from(group.querySelectorAll<HTMLElement>('[role="radio"]:not([disabled])'));
		const index = radios.indexOf(current);
		const next = radios[(index + step + radios.length) % radios.length];
		next.focus();
		next.click();
	}
</script>

<button
	type="button"
	role="radio"
	aria-checked={checked}
	tabindex={tabbable ? 0 : -1}
	class="relative flex w-full flex-col gap-1.5 rounded-lg border py-3.5 pl-11 pr-4 text-left transition-colors duration-100 {checked
		? 'border-signal bg-signal/10'
		: 'border-line-strong bg-surface-1 hover:border-line-hover'} {className}"
	onclick={onselect}
	onkeydown={handleKeydown}
>
	<span
		class="absolute left-3.5 top-4 h-[18px] w-[18px] rounded-full border-[1.5px] {checked
			? 'border-signal'
			: 'border-line-hover'}"
		aria-hidden="true"
	>
		{#if checked}
			<span class="absolute inset-[3px] rounded-full bg-signal"></span>
		{/if}
	</span>
	<span class="flex flex-wrap items-center gap-2 text-md font-semibold text-fg">
		{title}
		{#if badge}
			<Badge variant="signal" size="sm" class="font-mono uppercase tracking-[0.04em]">{badge}</Badge>
		{/if}
	</span>
	{#if description}
		<span class="text-base text-fg-muted">{description}</span>
	{/if}
	{@render children?.()}
</button>
