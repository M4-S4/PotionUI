<script lang="ts">
	import { Badge, Button } from '$lib/components/ui';
	import { api } from '$lib/services/api';
	import { logger } from '$lib/utils/logger';
	import type { HttpRequestTestResult } from '$lib/types/automations';

	let {
		config,
		automationId,
		nodeId
	}: {
		config: Record<string, any>;
		automationId?: string;
		nodeId: string;
	} = $props();

	let sample = $state('{}');
	let running = $state(false);
	let error = $state<string | null>(null);
	let result = $state<HttpRequestTestResult | null>(null);

	let bodyText = $derived(
		result === null
			? ''
			: typeof result.body === 'string'
				? result.body
				: JSON.stringify(result.body, null, 2)
	);

	async function run() {
		let event: Record<string, any>;
		try {
			const parsed = JSON.parse(sample || '{}');
			if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error();
			event = parsed;
		} catch {
			error = 'Sample event must be a JSON object.';
			result = null;
			return;
		}
		running = true;
		error = null;
		result = null;
		try {
			const response = await api.testHttpRequest({
				config,
				automation_id: automationId,
				node_id: nodeId,
				event
			});
			if (response.success && response.data) {
				result = response.data;
			} else {
				error = response.message || 'The request failed.';
			}
		} catch (caught) {
			logger.error('HTTP request test failed:', caught);
			error = 'Could not run the test request.';
		} finally {
			running = false;
		}
	}
</script>

<section class="space-y-2 border-t border-line pt-3" data-http-request-test>
	<h3 class="text-xs font-mono font-semibold uppercase tracking-wide text-fg-subtle">Test request</h3>
	<label class="block">
		<span class="label">Sample event (JSON)</span>
		<textarea
			class="input textarea font-mono text-xs"
			rows="3"
			spellcheck="false"
			bind:value={sample}
			aria-label="Sample event"
		></textarea>
	</label>
	<Button size="sm" variant="secondary" icon="play" loading={running} onclick={run}>
		Test request
	</Button>
	{#if error}
		<p class="text-xs text-danger" role="alert" data-http-test-error>{error}</p>
	{/if}
	{#if result}
		<div class="space-y-1.5" data-http-test-result>
			<div class="flex items-center gap-2">
				<Badge variant={result.ok ? 'success' : 'danger'}>{result.status}</Badge>
				{#if result.truncated}
					<span class="text-xs text-warning">Body truncated at 1 MB</span>
				{/if}
			</div>
			<pre
				class="max-h-64 overflow-auto rounded border border-line bg-surface-2 p-2 font-mono text-xs text-fg whitespace-pre-wrap break-all"
				data-http-test-body>{bodyText}</pre>
		</div>
	{/if}
</section>
