import { describe, expect, it } from 'vitest';
import type { MediaToolRunState } from '$lib/media-tools/runs';
import { activeRuns, isRunChipRoute } from './activeRuns';

function run(key: string, runId: string, status: MediaToolRunState['status'] = 'running'): MediaToolRunState {
	return {
		key,
		runId,
		toolId: 'trim',
		label: 'Trim',
		progress: 0.5,
		status,
		message: null,
		outputMode: 'new',
		cancelling: false
	};
}

describe('activeRuns', () => {
	it('lists a run once even when it is keyed by several items', () => {
		const list = activeRuns({ a: run('a', 'r1'), b: run('b', 'r1'), c: run('c', 'r2') });
		expect(list.map((entry) => entry.runId)).toEqual(['r1', 'r2']);
	});

	it('leaves out finished runs', () => {
		expect(activeRuns({ a: run('a', 'r1', 'completed'), b: run('b', 'r2', 'failed') })).toEqual([]);
	});

	it('is empty with no runs', () => {
		expect(activeRuns({})).toEqual([]);
	});
});

describe('isRunChipRoute', () => {
	it('is true on History and Library and their subpaths', () => {
		expect(isRunChipRoute('/history')).toBe(true);
		expect(isRunChipRoute('/library')).toBe(true);
		expect(isRunChipRoute('/library/x')).toBe(true);
	});

	it('is false elsewhere, including lookalike prefixes', () => {
		expect(isRunChipRoute('/generate')).toBe(false);
		expect(isRunChipRoute('/library-extra')).toBe(false);
	});
});
