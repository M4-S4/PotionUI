<script lang="ts">
	import { tick } from 'svelte';
	import { slide } from 'svelte/transition';
	import { chatModes } from '$lib/stores/chatModes';
	import { chatToolRendererRegistry } from '$lib/registries/chatToolRendererRegistry';
	import { getApiErrorMessage, logger } from '$lib/utils/logger';
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import type { ToolExecution } from '$lib/types/chat';
	import {
		ACTION_STATE_LABEL,
		actionCardIcon,
		actionCardState,
		actionToolLabel,
		approveVerb,
		collapsedSummary,
		type ActionDecision,
		type ActionResolveOutcome
	} from '$lib/chat/actionCard';
	import type { VariableRoll, VariablesMap } from '$lib/utils/variableDefs';
	import ChatActionPreview from './ChatActionPreview.svelte';
	import Icon from '$lib/components/Icon.svelte';

	let {
		execution,
		actionKey,
		locked = false,
		externalError = null,
		variables = undefined,
		rolls = undefined,
		onResolve,
		onAskRedo
	}: {
		execution: ToolExecution;
		actionKey: string;
		locked?: boolean;
		externalError?: string | null;
		variables?: VariablesMap;
		rolls?: Record<string, VariableRoll>;
		onResolve?: (decision: ActionDecision) => Promise<ActionResolveOutcome>;
		onAskRedo?: () => void;
	} = $props();

	let busy = $state<'approve' | 'reject' | null>(null);
	let error = $state<string | null>(null);
	let note = $state<string | null>(null);
	let stale = $state(false);
	let rejecting = $state(false);
	let reasonText = $state('');
	let detailsOpen = $state(false);
	let pluginRenderer = $state<any>(null);
	let reasonInput = $state<HTMLInputElement | undefined>();

	const cardState = $derived(actionCardState(execution, stale));
	const toolInfo = $derived($chatModes.toolsCatalog.find((t) => t.name === execution.tool_name) ?? null);
	const toolLabel = $derived(actionToolLabel(execution.tool_name, $chatModes.toolsCatalog));
	const toolMeta = $derived(toolInfo ? { icon: toolInfo.icon ?? null, label: toolInfo.label ?? null } : null);
	const decidable = $derived(cardState === 'pending' || cardState === 'stale');
	const bodyOpen = $derived(decidable || detailsOpen);
	const canAct = $derived(!!onResolve && !locked && !busy);
	const verb = $derived(cardState === 'stale' ? 'Approve anyway' : approveVerb(execution));
	const summary = $derived(collapsedSummary(execution, toolLabel));
	const failureText = $derived(execution.result?.error || 'The action failed without an error message.');

	$effect(() => {
		const name = execution.tool_name;
		if (!chatToolRendererRegistry.has(name)) {
			pluginRenderer = null;
			return;
		}
		let cancelled = false;
		chatToolRendererRegistry
			.resolve(name)
			.then((component) => {
				if (!cancelled) pluginRenderer = component;
			})
			.catch((err) => logger.error('Failed to load the tool preview renderer:', err));
		return () => {
			cancelled = true;
		};
	});

	async function decide(decision: ActionDecision, kind: 'approve' | 'reject') {
		if (!canAct || !onResolve) return;
		busy = kind;
		error = null;
		note = null;
		try {
			const outcome = await onResolve(decision);
			stale = outcome.status === 'stale';
			note = outcome.note ?? null;
			if (!stale) {
				rejecting = false;
				reasonText = '';
				detailsOpen = false;
			}
		} catch (err) {
			error = getApiErrorMessage(err, 'Could not resolve this action.');
		} finally {
			busy = null;
		}
	}

	function approve() {
		void decide(cardState === 'stale' ? { approved: true, force: true } : { approved: true }, 'approve');
	}

	function retry() {
		void decide({ approved: true }, 'approve');
	}

	function confirmReject() {
		const reason = reasonText.trim();
		void decide(reason ? { approved: false, reason } : { approved: false }, 'reject');
	}

	async function openReject() {
		if (!canAct) return;
		rejecting = true;
		await tick();
		reasonInput?.focus();
	}

	function cancelReject() {
		rejecting = false;
		reasonText = '';
	}

	function onCardKeydown(event: KeyboardEvent) {
		if (event.key !== 'Enter' || event.target !== event.currentTarget) return;
		if (!decidable || rejecting) return;
		event.preventDefault();
		approve();
	}

	function enterApproves(node: HTMLElement, focusable: boolean) {
		node.tabIndex = focusable ? 0 : -1;
		node.addEventListener('keydown', onCardKeydown);
		return {
			update(next: boolean) {
				node.tabIndex = next ? 0 : -1;
			},
			destroy() {
				node.removeEventListener('keydown', onCardKeydown);
			}
		};
	}

	function onReasonKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter') {
			event.preventDefault();
			confirmReject();
		} else if (event.key === 'Escape') {
			event.preventDefault();
			cancelReject();
		}
	}
</script>

{#if cardState}
	<div
		class="action-card is-{cardState}"
		data-testid="action-card"
		data-action-key={actionKey}
		data-state={cardState}
		role="group"
		aria-label="{toolLabel}: {ACTION_STATE_LABEL[cardState]}"
		use:enterApproves={decidable}
	>
		<div class="action-card-head">
			<div class="action-card-title">
				<Icon name={actionCardIcon(execution)} className="icon" strokeWidth={1.8} />
				<strong>{toolLabel}</strong>
				<span class="action-label is-{cardState}">{ACTION_STATE_LABEL[cardState]}</span>
			</div>
			<div class="action-card-actions">
				{#if decidable}
					<button
						type="button"
						class="action-btn is-reject"
						aria-pressed={rejecting}
						disabled={!canAct}
						onclick={openReject}
					>
						{#if busy === 'reject'}<Spinner size="sm" />{/if}Reject
					</button>
					<button
						type="button"
						class="action-btn is-approve"
						disabled={!canAct || rejecting}
						onclick={approve}
					>
						{#if busy === 'approve'}<Spinner size="sm" />{/if}{verb}{#if cardState === 'pending'}<span
								class="action-kbd"
								aria-hidden="true">↵</span
							>{/if}
					</button>
				{:else if cardState === 'failed'}
					<button type="button" class="action-btn is-ghost" disabled={!canAct} onclick={retry}>
						{#if busy === 'approve'}<Spinner size="sm" />{/if}Retry
					</button>
				{:else}
					<button
						type="button"
						class="action-btn is-ghost"
						aria-expanded={detailsOpen}
						onclick={() => (detailsOpen = !detailsOpen)}
					>
						{detailsOpen ? 'Hide details' : 'Details'}
					</button>
				{/if}
			</div>
		</div>

		{#if rejecting && decidable}
			<div class="action-reason" transition:slide={{ duration: 150 }}>
				<input
					type="text"
					bind:this={reasonInput}
					bind:value={reasonText}
					maxlength={2000}
					placeholder="Reason (optional)"
					aria-label="Reason (optional)"
					onkeydown={onReasonKeydown}
				/>
				<button type="button" class="action-btn is-ghost" disabled={!!busy} onclick={cancelReject}>Cancel</button>
				<button type="button" class="action-btn is-reject" disabled={!canAct} onclick={confirmReject}>
					Reject
				</button>
			</div>
		{/if}

		{#if cardState === 'stale'}
			<div class="action-stale">
				<span>You changed the form after this was proposed.</span>
				<button type="button" class="action-btn is-ghost" disabled={locked || !!busy} onclick={() => onAskRedo?.()}>
					Ask to redo
				</button>
			</div>
		{/if}

		{#if cardState === 'failed'}
			<div class="action-error" data-testid="action-error">{failureText}</div>
		{/if}

		{#if cardState === 'applied' && !detailsOpen}
			<div class="action-summary">{summary}</div>
		{/if}

		{#if cardState === 'rejected' && !detailsOpen}
			<div class="action-summary" class:is-muted={!execution.approval_reason}>
				{execution.approval_reason ? `"${execution.approval_reason}"` : 'Rejected without a reason.'}
			</div>
		{/if}

		{#if cardState === 'expired' && !detailsOpen}
			<div class="action-summary is-muted">This approval expired before it was decided.</div>
		{/if}

		{#if bodyOpen}
			<div class="action-card-body" transition:slide={{ duration: 150 }}>
				{#if pluginRenderer}
					{@const Renderer = pluginRenderer}
					<Renderer {execution} {toolMeta} expanded={true} />
				{:else}
					<ChatActionPreview {execution} {variables} {rolls} />
				{/if}
			</div>
		{/if}

		{#if note}
			<div class="action-note">{note}</div>
		{/if}
		{#if error || externalError}
			<div class="action-note is-error" role="alert">{error || externalError}</div>
		{/if}
		{#if cardState === 'pending' && !rejecting}
			<div class="action-hint">Or reply below to reject it with your message as the reason.</div>
		{/if}
	</div>
{/if}
