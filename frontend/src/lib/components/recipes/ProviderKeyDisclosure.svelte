<script lang="ts">
	import { api } from '$lib/services/api/index';
	import type { SetupConsentProvider } from '$lib/services/api/setup';
	import { getApiErrorMessage } from '$lib/utils/logger';
	import { Button, Input } from '$lib/components/ui';

	let { provider, generic = false }: { provider: SetupConsentProvider; generic?: boolean } = $props();

	let draft = $state('');
	let busy = $state(false);
	let error = $state('');
	let saved = $state(false);

	const done = $derived(provider.configured || saved);

	async function save() {
		const value = draft.trim();
		if (!value || busy) return;
		busy = true;
		error = '';
		try {
			await api.saveSetupProviderCredential(provider.id, provider.field_name, value);
			saved = true;
			draft = '';
		} catch (err: any) {
			error = getApiErrorMessage(err, "Couldn't save the token.");
		} finally {
			busy = false;
		}
	}
</script>

{#if done}
	<p class="text-xs text-success" data-provider-key="saved">{provider.name} API key saved.</p>
{:else}
	<details class="rounded border border-line bg-surface-1 px-3 py-2" data-provider-key="form">
		<summary class="cursor-pointer text-sm font-medium text-fg-muted">
			{generic ? 'Add a token (optional)' : `Add a ${provider.name} token (optional)`}
		</summary>
		<div class="mt-2 space-y-2">
			<p class="text-xs text-fg-muted">
				Needed for gated models. Can help if downloads fail with rate-limit errors. Stored on this
				computer.
				{#if provider.website}
					<a href={provider.website} target="_blank" rel="noreferrer" class="text-signal hover:underline">
						Get a token
					</a>
				{/if}
			</p>
			<div class="flex items-center gap-2">
				<Input
					type="password"
					autocomplete="off"
					class="flex-1 text-sm"
					placeholder="Paste your token"
					bind:value={draft}
				/>
				<Button size="sm" variant="secondary" loading={busy} disabled={!draft.trim()} onclick={save}>
					Save
				</Button>
			</div>
			{#if error}
				<p class="text-xs text-danger" role="alert">{error}</p>
			{/if}
		</div>
	</details>
{/if}
