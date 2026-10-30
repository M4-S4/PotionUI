<script lang="ts">
	import Icon from '$lib/components/Icon.svelte';
	import { TRUST_LINE } from '$lib/utils/setupGuide';
</script>

<footer class="border-t border-line bg-surface-1">
	<div class="mx-auto flex max-w-[720px] items-start gap-2.5 px-6 py-3 text-base text-fg-muted max-sm:px-4">
		<Icon name="monitor" className="mt-0.5 h-4 w-4 flex-none text-fg-subtle" />
		<span>{TRUST_LINE}</span>
	</div>
</footer>
