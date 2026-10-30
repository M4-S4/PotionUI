import type { MediaToolRunState } from '$lib/media-tools/runs';
import type { GenerationHistoryItem } from '$lib/types/history';

export function runForGeneration(
	runs: Record<string, MediaToolRunState>,
	generation: Pick<GenerationHistoryItem, 'id' | 'files'>
): MediaToolRunState | null {
	const files = generation.files ?? [];
	for (let index = 0; index < files.length; index += 1) {
		const run = runs[`${generation.id}:${index}`];
		if (run) return run;
	}
	return null;
}
