import { isTerminalStatus } from '$lib/media-tools/types';
import type { MediaToolRunState } from '$lib/media-tools/runs';

const CHIP_ROUTES = ['/history', '/library'];

export function activeRuns(runs: Record<string, MediaToolRunState>): MediaToolRunState[] {
	const seen = new Set<string>();
	const list: MediaToolRunState[] = [];
	for (const run of Object.values(runs)) {
		if (seen.has(run.runId) || isTerminalStatus(run.status)) continue;
		seen.add(run.runId);
		list.push(run);
	}
	return list;
}

export function isRunChipRoute(pathname: string): boolean {
	return CHIP_ROUTES.some((route) => pathname === route || pathname.startsWith(`${route}/`));
}
