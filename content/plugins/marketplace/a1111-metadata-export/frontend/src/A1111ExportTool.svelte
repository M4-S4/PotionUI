<script>
	import HostUi from '../../../../sdk/HostUi.svelte';

	let { generationIds = [], generations = [], onClose } = $props();

	let exporting = $state(false);
	let errorMessage = $state('');

	function isImageFile(file) {
		return (file?.file_type || '').toUpperCase() === 'IMAGE';
	}

	let generationCount = $derived(generationIds.length);
	let imageCount = $derived(
		generations.reduce((total, generation) => total + (generation.files || []).filter(isImageFile).length, 0)
	);

	function notify(level, message) {
		window.__potionui?.notifications?.toast(level, message);
	}

	async function errorFromResponse(response) {
		try {
			const payload = await response.json();
			const detail = payload?.detail;
			if (typeof detail === 'string') return detail;
			if (detail?.message) return detail.message;
		} catch {
			return `Export failed (${response.status})`;
		}
		return `Export failed (${response.status})`;
	}

	async function download() {
		if (exporting || generationIds.length === 0) return;
		exporting = true;
		errorMessage = '';
		try {
			const response = await window.__potionui.api.fetch('/api/plugins/a1111-metadata-export/export-zip', {
				method: 'POST',
				timeout: 0,
				body: JSON.stringify({ generation_ids: generationIds })
			});
			if (!response.ok) throw new Error(await errorFromResponse(response));

			const blob = await response.blob();
			const url = URL.createObjectURL(blob);
			const anchor = document.createElement('a');
			anchor.href = url;
			anchor.download = `a1111-export-${generationIds.length}.zip`;
			document.body.appendChild(anchor);
			anchor.click();
			anchor.remove();
			URL.revokeObjectURL(url);

			notify('success', 'A1111 metadata export downloaded');
			onClose?.();
		} catch (e) {
			errorMessage = e instanceof Error ? e.message : 'Export failed.';
			notify('error', 'A1111 metadata export failed');
		} finally {
			exporting = false;
		}
	}
</script>

<div class="body">
	<p class="summary">
		{generationCount} generation{generationCount === 1 ? '' : 's'}, {imageCount} image{imageCount === 1 ? '' : 's'} to
		export.
	</p>
	<p class="hint">A1111, Forge and CivitAI read the embedded parameters chunk on upload and auto-fill the prompt, sampler, model, and LoRAs.</p>

	{#if errorMessage}
		<HostUi name="Alert" props={{ variant: 'danger', density: 'compact', icon: true }}>{errorMessage}</HostUi>
	{/if}

	<div class="actions">
		<HostUi name="Button" props={{ variant: 'secondary', size: 'sm', onclick: onClose, disabled: exporting }}>Cancel</HostUi>
		<HostUi
			name="Button"
			props={{ variant: 'primary', size: 'sm', onclick: download, loading: exporting, disabled: exporting || imageCount === 0 }}
		>
			Download ZIP
		</HostUi>
	</div>
</div>

<style>
	.body {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}
	.summary {
		margin: 0;
		font-size: 13px;
		font-variant-numeric: tabular-nums;
		color: rgb(var(--fg, 232 234 237));
	}
	.hint {
		margin: 0;
		font-size: 12px;
		color: rgb(var(--fg-muted, 169 174 184));
	}
	.actions {
		display: flex;
		justify-content: flex-end;
		gap: 8px;
		margin-top: 4px;
	}
</style>
