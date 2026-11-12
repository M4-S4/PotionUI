<script>
	import { formatNumber, t } from '../../i18n/index.svelte.js';
	import { doc, act, relayout } from '../../lib/state.svelte.js';
	import { DISSOLVE_CHOICES, asked, dissolveBefore, lostSeconds, setDissolve } from '../../lib/dissolve.js';

	let { clip } = $props();

	const id = $props.id();
	const index = $derived(doc.clips.findIndex((c) => c.id === clip?.id));
	const wanted = $derived(asked(clip));
	const applied = $derived(index > 0 ? dissolveBefore(doc.clips, index) : 0);
	const choices = $derived(wanted && !DISSOLVE_CHOICES.includes(wanted) ? [...DISSOLVE_CHOICES, wanted].sort((a, b) => a - b) : DISSOLVE_CHOICES);
	const total = $derived(lostSeconds(doc.clips));
	const value = (v) => formatNumber(v, { maximumFractionDigits: 2 });
	const sec = (v) => `${formatNumber(v, { minimumFractionDigits: 1, maximumFractionDigits: 2 })} s`;

	function choose(v) {
		act(() => {
			if (setDissolve(doc.clips, index, v)) relayout();
		});
	}
</script>

{#if index > 0}
	<div class="flex w-full flex-col gap-1.5" data-seam-transition={clip.id}>
		<div class="flex items-center gap-2">
			<label for={id} class="w-24 shrink-0 text-xs text-fg-muted">{t('timeline.dissolve.label')}</label>
			<select {id} class="input vp-dense min-w-0 flex-1" value={String(wanted)} onchange={(e) => choose(+e.currentTarget.value)}>
				<option value="0">{t('timeline.dissolve.cut')}</option>
				{#each choices as v (v)}<option value={String(v)}>{t('timeline.dissolve.value', { value: value(v) })}</option>{/each}
			</select>
		</div>
		{#if applied > 0}
			<p class="text-xs text-warning" data-dissolve-loss>{t('timeline.dissolve.shorter', { dur: sec(applied) })}</p>
			{#if applied < wanted}<p class="text-xs text-fg-subtle" data-dissolve-capped>{t('timeline.dissolve.capped', { dur: sec(applied) })}</p>{/if}
			{#if total > applied}<p class="text-xs text-fg-subtle" data-dissolve-total>{t('timeline.dissolve.total', { dur: sec(total) })}</p>{/if}
		{:else if wanted > 0}
			<p class="text-xs text-fg-subtle" data-dissolve-capped>{t('timeline.dissolve.tooShort')}</p>
		{:else}
			<p class="text-xs text-fg-subtle">{t('timeline.dissolve.hint')}</p>
		{/if}
	</div>
{/if}
