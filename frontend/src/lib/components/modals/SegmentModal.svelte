<script module lang="ts">
	export type SegmentModalTier = 'S' | 'M' | 'L';
	export type SegmentModalTone = 'condition' | 'variable' | 'phrase' | 'choice' | 'resource' | 'syntax';
</script>

<script lang="ts">
	import type { Snippet } from 'svelte';
	import BaseModal from './BaseModal.svelte';
	import SegmentModal from './SegmentModal.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import { Button, Kbd } from '$lib/components/ui';
	import { topDismissLayer } from '$lib/actions/layerStack';
	import { primaryModifier } from '$lib/utils/platform';
	import { createConfirmSettlementGate } from './confirmKeyboard';

	let {
		isOpen = true,
		tier,
		title,
		titleId,
		context = '',
		contextContent,
		mark = '',
		tone,
		icon = '',
		primaryLabel,
		primaryVariant = 'primary',
		primaryDisabled = false,
		busy = false,
		cancelLabel = 'Cancel',
		hideCancel = false,
		submit = 'enter',
		keyboard = true,
		dirty = false,
		discardMessage = 'Your edits are lost.',
		role = 'dialog',
		testId,
		footerStart,
		children,
		onConfirm,
		onCancel
	}: {
		isOpen?: boolean;
		tier: SegmentModalTier;
		title: string;
		titleId?: string;
		context?: string;
		contextContent?: Snippet;
		mark?: string;
		tone?: SegmentModalTone;
		icon?: string;
		primaryLabel: string;
		primaryVariant?: 'primary' | 'danger';
		primaryDisabled?: boolean;
		busy?: boolean;
		cancelLabel?: string;
		hideCancel?: boolean;
		submit?: 'enter' | 'ctrl';
		keyboard?: boolean;
		dirty?: boolean;
		discardMessage?: string;
		role?: 'dialog' | 'alertdialog';
		testId?: string;
		footerStart?: Snippet;
		children?: Snippet;
		onConfirm: () => unknown;
		onCancel: () => void;
	} = $props();

	const TIER_CLASSES: Record<SegmentModalTier, string> = {
		S: 'md:w-[420px] md:max-w-[calc(100vw-3rem)]',
		M: 'md:w-[600px] md:max-w-[calc(100vw-3rem)]',
		L: 'md:w-[1040px] md:max-w-[calc(100vw-3rem)] md:h-[min(760px,calc(100dvh-4rem))]'
	};

	const TONE_CLASSES: Record<SegmentModalTone, string> = {
		condition: 'text-ai-3',
		variable: 'text-ai-1',
		phrase: 'text-signal',
		choice: 'text-warning',
		resource: 'text-success',
		syntax: 'text-warning'
	};

	const uid = Math.random().toString(36).slice(2, 8);
	const headingId = $derived(titleId ?? `segment-modal-title-${uid}`);
	const gate = createConfirmSettlementGate();

	let bodyRef = $state<HTMLDivElement>();
	let confirmingDiscard = $state(false);

	$effect(() => {
		if (isOpen) gate.reset();
		else confirmingDiscard = false;
	});

	function isTop(): boolean {
		const top = topDismissLayer();
		return !!top && !!bodyRef && top.node.contains(bodyRef);
	}

	function confirm() {
		if (!isOpen || busy || primaryDisabled) return;
		gate.settle(() => {
			let result: unknown;
			try {
				result = onConfirm();
			} catch {
				gate.reset();
				return;
			}
			if (result instanceof Promise) {
				result.then(
					(value) => {
						if (value === false) gate.reset();
					},
					() => gate.reset()
				);
			} else if (result === false) {
				gate.reset();
			}
		});
	}

	function requestCancel() {
		if (!isOpen || busy) return;
		if (dirty) {
			confirmingDiscard = true;
			return;
		}
		gate.settle(onCancel);
	}

	function discard() {
		confirmingDiscard = false;
		gate.settle(onCancel);
	}

	const INTERACTIVE = 'textarea, select, button, a[href], summary, [contenteditable="true"], [role="button"], [role="option"], [role="menuitem"], [role="radio"], [role="tab"], [role="checkbox"], [role="switch"]';
	const SINGLE_LINE = new Set(['text', 'search', 'email', 'url', 'tel', 'number', 'password', '']);

	function enterSubmits(target: EventTarget | null): boolean {
		if (!(target instanceof HTMLElement)) return true;
		if (target instanceof HTMLInputElement) {
			return SINGLE_LINE.has(target.type) && target.getAttribute('aria-expanded') !== 'true';
		}
		return !target.closest(INTERACTIVE);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (!isOpen || !keyboard || e.key !== 'Enter' || e.isComposing || e.defaultPrevented || confirmingDiscard || !isTop()) return;
		if (e.ctrlKey || e.metaKey) {
			e.preventDefault();
			if (!e.repeat) confirm();
			return;
		}
		if (submit !== 'enter' || e.shiftKey || e.altKey || !enterSubmits(e.target)) return;
		e.preventDefault();
		if (!e.repeat) confirm();
	}
</script>

<svelte:window onkeydowncapture={handleKeydown} />

<BaseModal
	{isOpen}
	sizeClass={TIER_CLASSES[tier]}
	compact={tier === 'S'}
	dialogRole={role}
	labelledBy={headingId}
	closeable={!busy}
	handleEscapeKey={keyboard}
	on:close={requestCancel}
>
	<svelte:fragment slot="headerIcon">
		<span
			class="grid h-7 w-7 flex-shrink-0 place-items-center rounded border border-line bg-surface-2 font-mono text-base font-semibold leading-none {tone
				? TONE_CLASSES[tone]
				: 'text-fg-muted'}"
			aria-hidden="true"
			data-segment-modal-mark
		>
			{#if mark}{mark}{:else if icon}<Icon name={icon} className="h-4 w-4" />{/if}
		</span>
	</svelte:fragment>
	<svelte:fragment slot="header">
		<div class="min-w-0 flex-1" role="heading" aria-level="2">
			<strong id={headingId} class="block truncate text-md font-semibold text-fg">{title}</strong>
			{#if contextContent}
				<span class="block truncate text-sm text-fg-muted" data-segment-modal-context>{@render contextContent()}</span>
			{:else if context}
				<span class="block truncate text-sm text-fg-muted" data-segment-modal-context>{context}</span>
			{/if}
		</div>
	</svelte:fragment>

	<div bind:this={bodyRef} class={tier === 'L' ? 'h-full' : 'min-h-full'} data-segment-modal={tier} data-testid={testId}>
		{@render children?.()}
	</div>

	<svelte:fragment slot="footer">
		<div class="flex items-center gap-3 px-5 py-3 max-md:flex-wrap max-md:gap-2 max-md:px-4" data-segment-modal-footer>
			<div class="flex min-w-0 flex-1 flex-wrap items-center gap-1.5 empty:hidden max-md:basis-full" data-segment-modal-start>
				{@render footerStart?.()}
			</div>
			<div class="ml-auto flex flex-shrink-0 items-center gap-2 max-md:ml-0 max-md:flex-1" data-segment-modal-end>
				{#if !hideCancel}
					<Button
						variant="ghost"
						size="sm"
						class="max-md:h-11 max-md:flex-1 max-md:bg-surface-3 max-md:text-fg"
						disabled={busy}
						initialFocus={primaryVariant === 'danger'}
						onclick={requestCancel}
					>
						<span class="inline-flex items-center gap-2">
							{cancelLabel}
							<span class="max-md:hidden"><Kbd keys="Esc" /></span>
						</span>
					</Button>
				{/if}
				<Button
					variant={primaryVariant}
					size="sm"
					class="max-md:h-11 max-md:flex-1"
					disabled={primaryDisabled || busy}
					loading={busy}
					initialFocus={submit === 'enter' && primaryVariant !== 'danger'}
					onclick={confirm}
				>
					<span class="inline-flex items-center gap-2" data-segment-modal-primary>
						{primaryLabel}
						<span class="max-md:hidden"><Kbd keys={submit === 'ctrl' ? [primaryModifier(), 'Enter'] : 'Enter'} /></span>
					</span>
				</Button>
			</div>
		</div>
	</svelte:fragment>
</BaseModal>

{#if confirmingDiscard}
	<SegmentModal
		tier="S"
		role="alertdialog"
		icon="warning"
		tone="syntax"
		title="Discard changes?"
		context={title}
		primaryLabel="Discard"
		primaryVariant="danger"
		cancelLabel="Keep editing"
		onConfirm={discard}
		onCancel={() => (confirmingDiscard = false)}
	>
		<p class="px-5 py-4 text-base text-fg-muted">{discardMessage}</p>
	</SegmentModal>
{/if}
