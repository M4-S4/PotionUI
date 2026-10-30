<script lang="ts">
	import { SETUP_STEP_COUNT, stepIndex, type SetupStepId, type StepStatus } from '$lib/utils/setupGuide';

	let {
		step,
		title,
		lede,
		status = null,
		titleId
	}: {
		step: SetupStepId;
		title: string;
		lede?: string;
		status?: StepStatus | null;
		titleId?: string;
	} = $props();

	const toneClass = {
		signal: 'text-signal',
		success: 'text-success',
		warning: 'text-warning'
	};
</script>

<div>
	<p class="font-mono text-sm uppercase tracking-[0.06em] text-fg-subtle">
		Step {stepIndex(step) + 1} of {SETUP_STEP_COUNT}
		{#if status}
			<span class={toneClass[status.tone]}>- {status.text}</span>
		{/if}
	</p>
	<h2 id={titleId} class="mt-1.5 text-[26px] font-semibold leading-8 tracking-[-0.02em] text-fg max-sm:text-2xl">
		{title}
	</h2>
	{#if lede}
		<p class="mt-2 max-w-[60ch] text-lg text-fg-muted">{lede}</p>
	{/if}
</div>
