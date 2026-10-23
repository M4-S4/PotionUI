<script lang="ts" generics="T extends CollectionLike">
	import { hasSubcollections } from './subcollections';
	import type { CollectionLike } from './types';

	let {
		collections,
		activeId,
		directOnly,
		onDirectOnlyChange
	}: {
		collections: T[];
		activeId: string | undefined;
		directOnly: boolean;
		onDirectOnlyChange: (directOnly: boolean) => void;
	} = $props();

	let visible = $derived(hasSubcollections(collections, activeId));
</script>

{#if visible}
	<div class="col-span-2" data-testid="include-subcollections">
		<span class="mb-1 block text-2xs uppercase tracking-[0.07em] text-fg-subtle">Subcollections</span>
		<label
			class="flex cursor-pointer select-none items-center gap-2 rounded bg-surface-3/50 px-2 py-1.5 text-xs text-fg-muted hover:text-fg"
		>
			<span class="relative inline-flex h-[15px] w-[15px] flex-shrink-0">
				<input
					type="checkbox"
					class="peer absolute inset-0 m-0 cursor-pointer opacity-0"
					checked={!directOnly}
					onchange={(e) => onDirectOnlyChange(!e.currentTarget.checked)}
				/>
				<span
					class="pointer-events-none absolute inset-0 rounded border border-line-hover bg-surface-2 transition-colors peer-checked:border-signal-solid peer-checked:bg-signal-solid peer-focus-visible:ring-2 peer-focus-visible:ring-signal peer-focus-visible:ring-offset-1 peer-focus-visible:ring-offset-canvas"
				></span>
				<svg
					class="pointer-events-none absolute inset-0 h-full w-full text-white opacity-0 transition-opacity peer-checked:opacity-100"
					viewBox="0 0 15 15"
					fill="none"
					aria-hidden="true"
				>
					<path
						d="M3 7.5L6.5 11L12 4.5"
						stroke="currentColor"
						stroke-width="1.5"
						stroke-linecap="round"
						stroke-linejoin="round"
					/>
				</svg>
			</span>
			Include subcollections
		</label>
	</div>
{/if}
