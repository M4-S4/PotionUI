<script lang="ts">
	import { onDestroy, untrack } from 'svelte';
	import { api } from '$lib/services/api/index';
	import type { LLMContextWindowDiscovery } from '$lib/types/llm';
	import { Button, Spinner } from '$lib/components/ui';
	import { contextWindowReadout } from '../llmProviderOptions';

	let {
		id,
		type,
		model,
		baseUrl,
		apiKey = '',
		configId = null,
		value,
		onchange,
		debounceMs = 400
	}: {
		id: string;
		type: string;
		model: string;
		baseUrl: string;
		apiKey?: string;
		configId?: string | null;
		value: number | undefined;
		onchange: (next: number | undefined) => void;
		debounceMs?: number;
	} = $props();

	let result = $state<LLMContextWindowDiscovery | null>(null);
	let requestError = $state<string | null>(null);
	let loading = $state(false);
	let requestSeq = 0;
	let debounceTimer: ReturnType<typeof setTimeout> | null = null;

	const readout = $derived(contextWindowReadout(result, value != null, requestError));

	async function detect() {
		if (debounceTimer) {
			clearTimeout(debounceTimer);
			debounceTimer = null;
		}
		const seq = ++requestSeq;
		loading = true;
		try {
			const response = await api.discoverLLMContextWindow({
				type,
				base_url: baseUrl.trim(),
				api_key: apiKey || null,
				config_id: configId,
				model
			});
			if (seq !== requestSeq) return;
			if (response.success && response.data) {
				result = response.data;
				requestError = null;
			} else {
				result = null;
				requestError = response.message || 'Failed to detect the context window.';
			}
		} catch {
			if (seq !== requestSeq) return;
			result = null;
			requestError = 'Failed to reach the server to detect the context window.';
		} finally {
			if (seq === requestSeq) loading = false;
		}
	}

	$effect(() => {
		void [type, model];
		if (debounceTimer) clearTimeout(debounceTimer);
		if (!model.trim()) {
			requestSeq += 1;
			result = null;
			requestError = null;
			loading = false;
			return;
		}
		debounceTimer = setTimeout(() => {
			debounceTimer = null;
			untrack(detect);
		}, debounceMs);
	});

	onDestroy(() => {
		requestSeq += 1;
		if (debounceTimer) clearTimeout(debounceTimer);
	});

	function onInput(e: Event) {
		const raw = (e.target as HTMLInputElement).value;
		const parsed = parseInt(raw, 10);
		onchange(raw === '' || !Number.isFinite(parsed) || parsed < 1 ? undefined : parsed);
	}
</script>

<div class="space-y-2">
	<input
		{id}
		type="number"
		class="input font-mono tabular-nums"
		min="1"
		step="1"
		value={value ?? ''}
		placeholder="Auto"
		oninput={onInput}
	/>
	<div class="flex items-center gap-3 min-h-7">
		<Button variant="secondary" size="xs" icon="refresh" disabled={loading || !model.trim()} onclick={detect}>
			Detect
		</Button>
		<div class="text-xs min-w-0 flex items-center gap-1.5 {readout?.tone === 'warning' ? 'text-warning' : 'text-fg-subtle'}" aria-live="polite" data-testid="context-window-readout">
			{#if loading}
				<Spinner size="sm" />
				<span>Detecting…</span>
			{:else if readout}
				<span>
					{#each readout.parts as part}{#if typeof part === 'string'}{part}{:else}<span class="font-mono tabular-nums">{part.n.toLocaleString()}</span>{/if}{/each}
				</span>
			{/if}
		</div>
	</div>
</div>
