import type { GenerationHistoryItem } from '$lib/types/history';

export const QUEUE_WAIT_MIN_SECONDS = 5;

function clock(totalSeconds: number): string {
	const hours = Math.floor(totalSeconds / 3600);
	const minutes = Math.floor((totalSeconds % 3600) / 60);
	const seconds = totalSeconds % 60;
	const ss = String(seconds).padStart(2, '0');
	if (hours > 0) return `${hours}:${String(minutes).padStart(2, '0')}:${ss}`;
	return `${minutes}:${ss}`;
}

export function formatGenerationTime(seconds: number | null | undefined): string | null {
	if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) return null;
	const whole = Math.round(seconds);
	if (whole < 1) return '<1 s';
	if (whole < 60) return `${whole} s`;
	return clock(whole);
}

export function generationRunLabel(
	generation: Pick<GenerationHistoryItem, 'status' | 'run_seconds' | 'completed_at'>
): string | null {
	if (generation.status !== 'completed' || !generation.completed_at) return null;
	return formatGenerationTime(generation.run_seconds);
}

export function generationQueueLabel(
	generation: Pick<GenerationHistoryItem, 'status' | 'queued_seconds' | 'completed_at'>
): string | null {
	if (generation.status !== 'completed' || !generation.completed_at) return null;
	const queued = generation.queued_seconds;
	if (typeof queued !== 'number' || !(queued >= QUEUE_WAIT_MIN_SECONDS)) return null;
	return formatGenerationTime(queued);
}

export function generationImageCount(generation: Pick<GenerationHistoryItem, 'files'>): number {
	return (generation.files ?? []).filter(
		(file) => file.is_final && !file.is_derived && file.file_type?.toLowerCase() === 'image'
	).length;
}

export function generationRunSummary(
	generation: Pick<GenerationHistoryItem, 'status' | 'run_seconds' | 'completed_at' | 'files'>
): string | null {
	const label = generationRunLabel(generation);
	if (!label) return null;
	const images = generationImageCount(generation);
	return images > 1 ? `${label} · ${images} images` : label;
}
