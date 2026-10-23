export function formatThinkingElapsed(ms: number): string {
	if (ms < 1000) return '<1s';
	const seconds = Math.round(ms / 1000);
	if (seconds < 60) return `${seconds}s`;
	const minutes = Math.floor(seconds / 60);
	const rest = seconds % 60;
	return `${minutes}m ${String(rest).padStart(2, '0')}s`;
}
