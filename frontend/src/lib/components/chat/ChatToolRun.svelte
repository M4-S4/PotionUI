<script lang="ts">
	import { slide } from 'svelte/transition';
	import type { ToolExecution } from '$lib/types/chat';
	import { chatModes } from '$lib/stores/chatModes';

	export let executions: ToolExecution[] = [];

	$: visible = executions.filter((e) => e.approval_status == null);

	function humanize(name: string): string {
		return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
	}

	function toolLabel(execution: ToolExecution): string {
		const info = $chatModes.toolsCatalog.find((t) => t.name === execution.tool_name);
		return info?.label || humanize(execution.tool_name);
	}

	function isFailed(execution: ToolExecution): boolean {
		return execution.status !== 'running' && !execution.skipped && !execution.result.success;
	}

	function formatDuration(ms: number): string {
		if (ms == null) return '';
		return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
	}

	$: anyFailed = visible.some(isFailed);
	$: namesLine = visible.map(toolLabel).join(' · ');
</script>

{#if visible.length > 0}
	<!-- Defaults open when a failed call needs surfacing, collapsed otherwise.
	     Markup ported verbatim from the mock's .tool-run (chat-rework BRIEF2). -->
	<details class="tool-run" class:failed={anyFailed} open={anyFailed}>
		<summary>
			<span class="tool-run-status"><svg class="icon"><use href={anyFailed ? '#i-close' : '#i-check'} /></svg></span>
			<strong>{visible.length} tool{visible.length === 1 ? '' : 's'} used</strong>
			<span class="tool-run-names">{namesLine}</span>
			<svg class="icon tool-run-chevron"><use href="#i-chevron" /></svg>
		</summary>
		<div class="tool-run-details" transition:slide={{ duration: 150 }}>
			{#each visible as execution}
				{@const failed = isFailed(execution)}
				{@const running = execution.status === 'running'}
				{@const skipped = !!execution.skipped}
				<div class="tool-execution" class:failed class:skipped>
					{#if running}
						<span class="tool-execution-spinner" role="status" aria-label="Running"></span>
					{:else if failed}
						<svg class="icon"><use href="#i-close" /></svg>
					{:else if skipped}
						<svg class="icon"><use href="#i-chevron" /></svg>
					{:else}
						<svg class="icon"><use href="#i-check" /></svg>
					{/if}
					<strong>{toolLabel(execution)}</strong>
					<span>{skipped ? 'Not run' : formatDuration(execution.duration_ms)}</span>
				</div>
				{#if failed && execution.result?.error}
					<div class="tool-execution-error" transition:slide={{ duration: 120 }}>{execution.result.error}</div>
				{/if}
			{/each}
		</div>
	</details>
{/if}
