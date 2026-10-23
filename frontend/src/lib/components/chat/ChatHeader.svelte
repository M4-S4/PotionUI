<script lang="ts">
	import { IconButton } from '$lib/components/ui';
	// Header: rail toggle · title (+ inline rename) · mode chip · context meter ·
	// model button · more menu · close. Markup ported verbatim from the mock's
	// .chat-header (chat-rework BRIEF2); the three dropdowns are plain
	// position:absolute children (mock's .floating-menu, positioned against
	// .conversation) — no portal, no computed fixed coordinates.
	import { tick } from 'svelte';
	import dismiss from '$lib/actions/dismiss';
	import { chatModes } from '$lib/stores/chatModes';
	import { chatSession, modeLocked } from '$lib/stores/chatSession';
	import ChatModeSelector from '$lib/components/chat/ChatModeSelector.svelte';
	import ConfirmModal from '$lib/components/modals/ConfirmModal.svelte';
	import Tooltip from '$lib/components/Tooltip.svelte';
	import { formatTokenCount } from '$lib/utils/chat';
	import Icon from '$lib/components/Icon.svelte';

	// Provider+Model (LLM config) selection
	export let llmConfigs: any[] = [];
	export let selectedConfigId = '';
	export let onSelectConfig: (id: string) => void;

	// Mode selection (locks once the conversation has messages)
	export let onSelectMode: (id: string) => void;
	/** The scope the current page resolves to (route or a declared override) —
	 * passed through to the mode chip so a locked conversation whose scope no
	 * longer matches the page can say so. Null while unresolved. */
	export let pageModeId: string | null = null;
	export let onNewChat: (() => void) | undefined = undefined;

	// Token usage: the last request's prompt-token count. No context-window
	// field exists on an LLM config today, so the ring has nothing to size an
	// arc against — it renders unfilled and the title carries the number.
	export let currentContextSize = 0;

	// History rail toggle (leftmost header control)
	export let railCollapsed = false;
	export let onToggleRail: () => void;

	// Conversation identity
	export let title = 'New conversation';
	/** e.g. "Reading <tab> · N images in thread" — blank renders nothing. */
	export let subtitle = '';
	/** False for a conversation that has no session yet — rename/export/delete need one. */
	export let hasSession = false;
	export let onRename: (newTitle: string) => void;
	export let onExportTranscript: () => void;
	export let onDeleteConversation: () => void;

	export let onClose: (() => void) | undefined = undefined;

	let showModelDropdown = false;
	let modelTriggerEl: HTMLButtonElement;

	let showMoreMenu = false;
	let moreTriggerEl: HTMLButtonElement;

	let renaming = false;
	let renameDraft = '';
	let renameInputEl: HTMLInputElement | undefined;

	let showDeleteConfirm = false;

	$: mode = $chatSession.mode;

	$: selectedModelName = llmConfigs.find((c) => c.id === selectedConfigId)?.name || 'Model';

	function toggleModelDropdown() {
		showModelDropdown = !showModelDropdown;
		if (showModelDropdown) showMoreMenu = false;
	}

	function toggleMoreMenu() {
		showMoreMenu = !showMoreMenu;
		if (showMoreMenu) showModelDropdown = false;
	}

	async function startRename() {
		if (!hasSession) return;
		showMoreMenu = false;
		renameDraft = title;
		renaming = true;
		await tick();
		renameInputEl?.focus();
		renameInputEl?.select();
	}

	function commitRename() {
		if (!renaming) return;
		renaming = false;
		const next = renameDraft.trim();
		if (next && next !== title) onRename(next);
	}

	function cancelRename() {
		renaming = false;
	}

	function handleRenameKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			commitRename();
		} else if (e.key === 'Escape') {
			e.preventDefault();
			cancelRename();
			e.stopPropagation();
		}
	}

	function handleExport() {
		showMoreMenu = false;
		onExportTranscript();
	}

	function openDeleteConfirm() {
		showMoreMenu = false;
		showDeleteConfirm = true;
	}

	function handleDeleteConfirmed() {
		showDeleteConfirm = false;
		onDeleteConversation();
	}
</script>

<header class="chat-header">
	<Tooltip text={railCollapsed ? 'Show history' : 'Hide history'} position="bottom">
		<IconButton
			icon="chat-menu"
			label={railCollapsed ? 'Show history' : 'Hide history'}
			size="sm"
			ariaPressed={!railCollapsed}
			onclick={onToggleRail}
		/>
	</Tooltip>

	{#if onNewChat}
		<Tooltip text="New chat" position="bottom">
			<IconButton icon="chat-plus" label="New chat" size="sm" onclick={onNewChat} />
		</Tooltip>
	{/if}

	<div class="header-title">
		{#if renaming}
			<input
				bind:this={renameInputEl}
				bind:value={renameDraft}
				type="text"
				on:blur={commitRename}
				on:keydown={handleRenameKeydown}
			/>
		{:else}
			<h1>{title}</h1>
		{/if}
		{#if subtitle && !renaming}
			<p>{subtitle}</p>
		{/if}
	</div>

	<div class="header-actions">
		{#if $chatModes.modes.length > 0}
			<ChatModeSelector
				modes={$chatModes.modes}
				selected={mode}
				locked={$modeLocked}
				onSelect={onSelectMode}
				{pageModeId}
				{onNewChat}
			/>
		{/if}

		{#if currentContextSize > 0}
			<Tooltip
				text={`Context: ${currentContextSize.toLocaleString()} tokens in the last request`}
				position="bottom"
			>
				<button type="button" class="context-meter">
					<span class="context-meter-ring"></span><span>{formatTokenCount(currentContextSize)}</span>
				</button>
			</Tooltip>
		{/if}

		{#if llmConfigs.length > 0}
			<Tooltip text="LLM Model" position="bottom">
				<button
					bind:this={modelTriggerEl}
					type="button"
					class="model-button"
					class:open={showModelDropdown}
					aria-expanded={showModelDropdown}
					data-testid="chat-header-model-trigger"
					on:click={toggleModelDropdown}
				>
					<span class="model-dot"></span>
					<span>{selectedModelName || 'Model'}</span>
					<Icon name="chat-chevron" className="icon chevron" strokeWidth={1.8} />
				</button>
			</Tooltip>
		{/if}

		<Tooltip text="Conversation options" position="bottom">
			<button
				bind:this={moreTriggerEl}
				type="button"
				aria-label="Conversation options"
				class="icon-button"
				aria-expanded={showMoreMenu}
				on:click={toggleMoreMenu}
			>
				<Icon name="chat-more" className="icon" strokeWidth={1.8} />
			</button>
		</Tooltip>

		{#if onClose}
			<div class="header-divider"></div>
			<Tooltip text="Close" kbd="Esc" position="bottom">
				<IconButton icon="chat-close" label="Close chat" size="sm" onclick={onClose} />
			</Tooltip>
		{/if}
	</div>

	{#if showModelDropdown && llmConfigs.length > 0}
		<div
			class="floating-menu"
			use:dismiss={{ onDismiss: () => (showModelDropdown = false), ignore: [modelTriggerEl] }}
			data-testid="chat-header-model-menu"
		>
			<div class="menu-label">Choose a model</div>
			{#each llmConfigs as config}
				<button
					type="button"
					class="model-option"
					class:selected={config.id === selectedConfigId}
					on:click={() => {
						onSelectConfig(config.id);
						showModelDropdown = false;
					}}
				>
					<span class="model-dot"></span>
					<span class="model-option-copy">
						<strong>{config.name}</strong>
						<small>{config.model}</small>
					</span>
					{#if config.id === selectedConfigId}
						<Icon name="chat-check" className="icon check" strokeWidth={1.8} />
					{/if}
				</button>
			{/each}
		</div>
	{/if}

	{#if showMoreMenu}
		<div
			class="floating-menu conversation-menu"
			use:dismiss={{ onDismiss: () => (showMoreMenu = false), ignore: [moreTriggerEl] }}
		>
			<button type="button" disabled={!hasSession} on:click={startRename}>
				<Icon name="chat-edit" className="icon" strokeWidth={1.8} />Rename conversation
			</button>
			<button type="button" disabled={!hasSession} on:click={handleExport}>
				<Icon name="chat-file" className="icon" strokeWidth={1.8} />Export transcript
			</button>
			<button type="button" class="danger-option" disabled={!hasSession} on:click={openDeleteConfirm}>
				<Icon name="chat-trash" className="icon" strokeWidth={1.8} />Delete conversation
			</button>
		</div>
	{/if}
</header>

<ConfirmModal
	isOpen={showDeleteConfirm}
	title="Delete this conversation?"
	message="This permanently deletes the conversation and its messages. This can't be undone."
	variant="danger"
	on:confirm={handleDeleteConfirmed}
	on:cancel={() => (showDeleteConfirm = false)}
/>
