import { describe, expect, it } from 'vitest';
import { runForGeneration } from './toolRuns';
import type { MediaToolRunState } from '$lib/media-tools/runs';

function run(key: string, extra: Partial<MediaToolRunState> = {}): MediaToolRunState {
	return {
		key,
		runId: `run-${key}`,
		toolId: 'speed',
		label: 'Speed',
		progress: 0.4,
		status: 'running',
		message: null,
		outputMode: 'new',
		cancelling: false,
		...extra
	};
}

const generation = { id: 'g1', files: [{ id: 1 }, { id: 2 }] } as never;

describe('runForGeneration', () => {
	it('finds a run keyed by the generation file index the menu used', () => {
		expect(runForGeneration({ 'g1:1': run('g1:1') }, generation)?.key).toBe('g1:1');
	});

	it('returns null when none of its files is running', () => {
		expect(runForGeneration({ 'g2:0': run('g2:0'), other: run('other') }, generation)).toBeNull();
	});

	it('returns null with no files', () => {
		expect(runForGeneration({ 'g1:0': run('g1:0') }, { id: 'g1', files: [] } as never)).toBeNull();
	});

	it('prefers the first file when several are running', () => {
		const runs = { 'g1:1': run('g1:1'), 'g1:0': run('g1:0') };
		expect(runForGeneration(runs, generation)?.key).toBe('g1:0');
	});
});
