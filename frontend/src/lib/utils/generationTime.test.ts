import { describe, expect, it } from 'vitest';
import {
	formatGenerationTime,
	generationQueueLabel,
	generationRunSummary
} from './generationTime';

const file = (over: Record<string, unknown> = {}) =>
	({ id: 1, file_path: 'a.png', file_type: 'IMAGE', is_final: true, created_at: '', ...over }) as never;

const done = (over: Record<string, unknown> = {}) =>
	({
		status: 'completed',
		completed_at: '2026-01-01T00:00:00Z',
		run_seconds: 92,
		queued_seconds: 0,
		files: [file()],
		...over
	}) as never;

describe('formatGenerationTime', () => {
	it.each([
		[0, '<1 s'],
		[8, '8 s'],
		[10, '10 s'],
		[48, '48 s'],
		[59.6, '1:00'],
		[92, '1:32'],
		[3599, '59:59'],
		[3725, '1:02:05']
	])('%s -> %s', (input, expected) => {
		expect(formatGenerationTime(input)).toBe(expected);
	});

	it.each([[null], [undefined], [-3], [NaN], [Infinity]])('%s -> nothing', (input) => {
		expect(formatGenerationTime(input as never)).toBeNull();
	});
});

describe('generationRunSummary', () => {
	it('shows the time alone for a single image', () => {
		expect(generationRunSummary(done())).toBe('1:32');
	});

	it('adds the image count for several images, ignoring derived and non-image files', () => {
		const files = [file(), file(), file({ is_derived: true }), file({ file_type: 'VIDEO' })];
		expect(generationRunSummary(done({ files }))).toBe('1:32 · 2 images');
	});

	it('shows <1 s for a run that rounds to zero', () => {
		expect(generationRunSummary(done({ run_seconds: 0 }))).toBe('<1 s');
	});

	it('shows nothing without timing, completion or when not completed', () => {
		expect(generationRunSummary(done({ run_seconds: null }))).toBeNull();
		expect(generationRunSummary(done({ run_seconds: undefined }))).toBeNull();
		expect(generationRunSummary(done({ completed_at: undefined }))).toBeNull();
		expect(generationRunSummary(done({ status: 'running' }))).toBeNull();
		expect(generationRunSummary(done({ status: 'failed' }))).toBeNull();
	});
});

describe('generationQueueLabel', () => {
	it('appears from 5 s of waiting', () => {
		expect(generationQueueLabel(done({ queued_seconds: 4 }))).toBeNull();
		expect(generationQueueLabel(done({ queued_seconds: 5 }))).toBe('5 s');
		expect(generationQueueLabel(done({ queued_seconds: 75 }))).toBe('1:15');
	});

	it('is absent when the stamp is missing', () => {
		expect(generationQueueLabel(done({ queued_seconds: null }))).toBeNull();
		expect(generationQueueLabel(done({ queued_seconds: undefined }))).toBeNull();
	});
});
