import { toasts } from '$lib/stores/toast';
import { rememberToolParams } from './catalog';
import { mediaToolRuns, type MediaToolRunOutcome, type MediaToolRunsStore, type RunHandle } from './runs';
import type {
	MediaToolInputRef,
	MediaToolItem,
	MediaToolOutputMode,
	MediaToolSource,
	MediaToolStep
} from './types';

export interface ExecuteOptions {
	label: string;
	steps: MediaToolStep[];
	sources: readonly MediaToolSource[];
	refs: MediaToolInputRef[];
	mode: MediaToolOutputMode;
	announce?: boolean;
	runs?: MediaToolRunsStore;
}

function nameOf(item: MediaToolItem | undefined): string {
	return item ? (item.original_filename ?? item.filename) : 'The result';
}

export function announceStart(label: string, mode: MediaToolOutputMode) {
	toasts.show('info', `${label} · ${mode === 'replace' ? 'Replaces the original.' : 'Saves as a new item.'}`, {
		title: 'Edit started'
	});
}

export function announceOutcome(outcome: MediaToolRunOutcome) {
	if (outcome.status === 'completed') {
		if (outcome.replaced) {
			toasts.show('success', `${nameOf(outcome.items[0])} was updated.`, { title: 'Original replaced' });
			return;
		}
		const message =
			outcome.items.length === 1
				? `${nameOf(outcome.items[0])} is in your library.`
				: `${outcome.items.length} items are in your library.`;
		toasts.show('success', message, { title: 'Saved as new' });
	} else if (outcome.status === 'failed') {
		toasts.show('error', outcome.message, { title: 'Edit failed' });
	} else {
		toasts.show('info', 'Nothing was saved.', { title: 'Edit cancelled' });
	}
}

export async function executeMediaTool(options: ExecuteOptions): Promise<RunHandle> {
	const runs = options.runs ?? mediaToolRuns;
	const handle = await runs.start({
		keys: options.sources.map((source) => source.id),
		toolId: options.steps[0].tool,
		label: options.label,
		body: {
			steps: options.steps,
			inputs: options.refs,
			output: { mode: options.mode }
		}
	});
	if (options.announce) announceStart(options.label, options.mode);

	const done = handle.done.then((outcome) => {
		if (outcome.status === 'completed' && options.steps.length === 1) {
			rememberToolParams(options.steps[0].tool, options.steps[0].params);
		}
		if (options.announce) announceOutcome(outcome);
		return outcome;
	});
	return { runId: handle.runId, done };
}
