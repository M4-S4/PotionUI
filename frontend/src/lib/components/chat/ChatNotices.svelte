<script lang="ts">
	import type { ChatNotice } from '$lib/types/chat';
	import Icon from '$lib/components/Icon.svelte';
	import { noticeAppearance } from '$lib/chat/noticeKinds';

	let { notices = [] }: { notices?: ChatNotice[] } = $props();
</script>

{#if notices.length > 0}
	<div class="mb-3 flex w-[min(100%,500px)] flex-col gap-1.5" role="status" aria-live="polite" data-testid="chat-notices">
		{#each notices as notice, index (`${notice.kind}:${index}`)}
			{@const look = noticeAppearance(notice.kind)}
			<div
				class="flex items-start gap-2 rounded border border-line bg-surface-1 px-3 py-2 text-sm text-fg-muted"
				data-testid="chat-notice"
				data-kind={notice.kind}
			>
				<Icon name={look.icon} className="mt-0.5 h-3.5 w-3.5 shrink-0 {look.tone}" strokeWidth={1.8} />
				<span class="min-w-0 break-words">{notice.message}</span>
			</div>
		{/each}
	</div>
{/if}
