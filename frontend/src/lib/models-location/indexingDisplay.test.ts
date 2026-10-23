import { describe, expect, it } from 'vitest';
import {
	comparedFilesText,
	digestBytesDone,
	digestPercent,
	digestPhase,
	fileCountText,
	formatEta,
	formatEtaAbout,
	formatRate,
	indexingDoneSummary,
	indexingIsActive,
	indexingIsRunning,
	indexingIsVisible,
	indexingPercent,
	indexingScanHasContent,
	isSetupPath,
	levelDescription,
	levelEstimateLine,
	levelProfile,
	levelTimeText,
	levelUsageText,
	setupModelsLine,
	smallFilesText,
	notIndexedTitle,
	pickSelectionSummary
} from './indexingDisplay';
import type { IndexingDigests, IndexingStatus } from '$lib/services/api/models';
import { GIB, LEVELS, MIB, digests } from '../../../tests/component/helpers/indexingFixtures';

function status(overrides: Partial<IndexingStatus> = {}): IndexingStatus {
	return { state: 'idle', ...overrides };
}

function withDigests(overrides: Partial<IndexingDigests> = {}, base: Partial<IndexingStatus> = {}): IndexingStatus {
	return status({ state: 'done', found_on_disk: 487, ...base, digests: digests(overrides) });
}

describe('indexingIsRunning', () => {
	it('is false without a status', () => {
		expect(indexingIsRunning(null)).toBe(false);
	});

	it('is true while scanning or indexing', () => {
		expect(indexingIsRunning(status({ state: 'scanning' }))).toBe(true);
		expect(indexingIsRunning(status({ state: 'indexing' }))).toBe(true);
	});

	it('is true when a restart is pending, even if the state itself looks terminal', () => {
		expect(indexingIsRunning(status({ state: 'done', restart_pending: true }))).toBe(true);
	});

	it('is false once idle/done/failed/cancelled/blocked with no restart pending', () => {
		for (const state of ['idle', 'done', 'failed', 'cancelled', 'blocked'] as const) {
			expect(indexingIsRunning(status({ state }))).toBe(false);
		}
	});
});

describe('indexingIsVisible', () => {
	it('is false without a status or while idle', () => {
		expect(indexingIsVisible(null)).toBe(false);
		expect(indexingIsVisible(status({ state: 'idle' }))).toBe(false);
	});

	it('stays visible while idle when files were skipped as duplicates', () => {
		expect(indexingIsVisible(status({ state: 'idle', skipped_duplicates_total: 2 }))).toBe(true);
		expect(indexingIsVisible(status({ state: 'idle', skipped_duplicates_total: 0 }))).toBe(false);
	});

	it('is true for every non-idle state', () => {
		for (const state of ['scanning', 'indexing', 'done', 'failed', 'cancelled', 'blocked'] as const) {
			expect(indexingIsVisible(status({ state }))).toBe(true);
		}
	});
});

describe('indexingPercent', () => {
	it('is null without a status, outside "indexing", or before the total is known', () => {
		expect(indexingPercent(null)).toBeNull();
		expect(indexingPercent(status({ state: 'scanning' }))).toBeNull();
		expect(indexingPercent(status({ state: 'indexing', processed: 0, total: 0 }))).toBeNull();
	});

	it('rounds processed/total to a 0-100 percent', () => {
		expect(indexingPercent(status({ state: 'indexing', processed: 47, total: 128 }))).toBe(37);
		expect(indexingPercent(status({ state: 'indexing', processed: 128, total: 128 }))).toBe(100);
	});
});

describe('indexingDoneSummary', () => {
	it('is null without a status or outside "done"', () => {
		expect(indexingDoneSummary(null)).toBeNull();
		expect(indexingDoneSummary(status({ state: 'indexing' }))).toBeNull();
	});

	it('is "zero" when nothing was found on disk', () => {
		expect(indexingDoneSummary(status({ state: 'done', found_on_disk: 0 }))).toEqual({ kind: 'zero' });
	});

	it('is "up_to_date" when everything found was already indexed', () => {
		expect(indexingDoneSummary(status({ state: 'done', found_on_disk: 908, indexed: 0 }))).toEqual({
			kind: 'up_to_date',
			found: 908
		});
	});

	it('is "new_indexed" and computes the already-indexed remainder when new files were indexed', () => {
		expect(indexingDoneSummary(status({ state: 'done', found_on_disk: 128, indexed: 20 }))).toEqual({
			kind: 'new_indexed',
			indexed: 20,
			alreadyIndexed: 108,
			found: 128
		});
	});
});

describe('indexingIsActive', () => {
	it('is false without a status or digests while the scan is finished', () => {
		expect(indexingIsActive(null)).toBe(false);
		expect(indexingIsActive(status({ state: 'done' }))).toBe(false);
	});

	it('is true while the scan runs, even without digests', () => {
		expect(indexingIsActive(status({ state: 'scanning' }))).toBe(true);
	});

	it('is true while files are queued and not user-paused, including waiting for a generation', () => {
		expect(indexingIsActive(withDigests({ queued: 3, pending_total: 3 }))).toBe(true);
		expect(indexingIsActive(withDigests({ queued: 3, pending_total: 3, pause_reason: 'generation' }))).toBe(true);
	});

	it('is false when the user paused, or when nothing is queued', () => {
		expect(indexingIsActive(withDigests({ queued: 3, pending_total: 3, paused: true }))).toBe(false);
		expect(indexingIsActive(withDigests({ queued: 0, deferred: 3, pending_total: 3 }))).toBe(false);
	});

	it('is true while a choice is awaited', () => {
		const awaiting = { awaiting_choice: true, files: 30, bytes: 1, min_files: 20, min_bytes: 1 };
		expect(indexingIsActive(withDigests({ plan: awaiting }))).toBe(true);
	});
});

describe('digestPhase', () => {
	it('is none without digests', () => {
		expect(digestPhase(null)).toBe('none');
		expect(digestPhase(status({ state: 'done' }))).toBe('none');
	});

	it('is choice while a choice is awaited, ahead of everything else', () => {
		const plan = { awaiting_choice: true, files: 30, bytes: 1, min_files: 20, min_bytes: 1 };
		expect(digestPhase(withDigests({ plan, pending_total: 30, deferred: 30, ready: 5, paused: true }))).toBe('choice');
	});

	it('is empty when there are no models at all', () => {
		expect(digestPhase(withDigests())).toBe('empty');
	});

	it('is not_started when files wait but none is queued, and up_to_date when nothing is pending', () => {
		expect(digestPhase(withDigests({ ready: 15, pending_total: 472, deferred: 472 }))).toBe('not_started');
		expect(digestPhase(withDigests({ ready: 487 }))).toBe('up_to_date');
	});

	it('is running while queued files are read', () => {
		expect(digestPhase(withDigests({ ready: 10, pending_total: 5, queued: 5 }))).toBe('running');
	});

	it('is waiting when the pause reason is a generation', () => {
		expect(digestPhase(withDigests({ ready: 10, pending_total: 5, queued: 5, pause_reason: 'generation' }))).toBe('waiting');
	});

	it('shows a user pause ahead of a generation pause', () => {
		expect(
			digestPhase(withDigests({ ready: 10, pending_total: 5, queued: 5, paused: true, pause_reason: 'generation' }))
		).toBe('paused');
	});
});

describe('digest progress', () => {
	it('is measured in bytes, counting the part of a file already read', () => {
		const d = digests({
			ready: 3,
			pending_total: 1,
			queued: 1,
			ready_bytes: 100,
			pending_bytes: 100,
			current: [{ model_id: 'm', filename: 'a', size: 100, done: 50 }]
		});
		expect(digestBytesDone(d)).toBe(150);
		expect(digestPercent(d)).toBe(75);
	});

	it('does not move when the file counts race ahead of the bytes', () => {
		const d = digests({ ready: 400, pending_total: 87, queued: 87, ready_bytes: 10, pending_bytes: 990 });
		expect(digestPercent(d)).toBe(1);
	});

	it('is 100 when nothing is pending and 0 when bytes are unknown but files wait', () => {
		expect(digestPercent(digests({ ready: 5 }))).toBe(100);
		expect(digestPercent(digests({ pending_total: 5 }))).toBe(0);
	});

	it('never exceeds 100', () => {
		const d = digests({
			ready_bytes: 100,
			pending_bytes: 10,
			current: [{ model_id: 'm', filename: 'a', size: 500, done: 500 }]
		});
		expect(digestPercent(d)).toBe(100);
	});
});

describe('formatEta', () => {
	it('is empty for missing or invalid values', () => {
		expect(formatEta(null)).toBe('');
		expect(formatEta(undefined)).toBe('');
		expect(formatEta(Number.NaN)).toBe('');
	});

	it('says under a minute below 45 seconds', () => {
		expect(formatEta(44)).toBe('under a minute');
		expect(formatEta(45)).toBe('1 min');
	});

	it('counts minutes up to 90 and then hours with minutes', () => {
		expect(formatEta(4 * 60)).toBe('4 min');
		expect(formatEta(89 * 60)).toBe('89 min');
		expect(formatEta(90 * 60)).toBe('1 h 30 m');
		expect(formatEta(5 * 3600 + 12 * 60)).toBe('5 h 12 m');
		expect(formatEta(5 * 3600)).toBe('5 h');
	});

	it('prefixes about except for the under-a-minute text', () => {
		expect(formatEtaAbout(5 * 3600 + 12 * 60)).toBe('about 5 h 12 m');
		expect(formatEtaAbout(10)).toBe('under a minute');
		expect(formatEtaAbout(null)).toBe('');
	});
});

describe('formatRate', () => {
	it('prints whole MB/s and 0 for no throughput', () => {
		expect(formatRate(96 * MIB)).toBe('96 MB/s');
		expect(formatRate(null)).toBe('0 MB/s');
		expect(formatRate(0)).toBe('0 MB/s');
	});
});

describe('levels', () => {
	const low = LEVELS[0];
	const normal = LEVELS[1];
	const max = LEVELS[2];

	it('looks a profile up by level', () => {
		expect(levelProfile(digests(), 'normal')).toEqual(normal);
		expect(levelProfile(null, 'low')).toBeNull();
	});

	it('describes each level from its profile', () => {
		expect(levelDescription(low)).toBe(
			'One worker at the lowest priority, reading at most 96 MB/s. Pauses while you generate.'
		);
		expect(levelDescription(normal)).toBe(
			'Two workers at a gentler-than-usual priority, up to 384 MB/s. Pauses while you generate.'
		);
		expect(levelDescription(max)).toBe(
			'Four workers at full speed, limited only by your disk. Does not pause for generations, so expect everything to feel slower.'
		);
	});

	it('says what a level uses', () => {
		expect(levelUsageText(low)).toBe('Uses at most one core and 96 MB/s.');
		expect(levelUsageText(normal)).toBe('Uses at most two cores and 384 MB/s.');
		expect(levelUsageText(max)).toBe('Uses up to four cores and as much of your disk as it can.');
	});

	it('follows the numbers the backend reports', () => {
		expect(levelDescription({ ...low, rate_bps: 50 * MIB })).toContain('at most 50 MB/s');
	});

	it('estimates the time at a level and falls back to the disk limit without a rate', () => {
		expect(levelTimeText(96 * MIB * 7200, low)).toBe('about 2 h');
		expect(levelTimeText(96 * MIB * 7200, max)).toBe('as fast as your disk allows');
		expect(levelTimeText(1, null)).toBe('');
	});

	it('words the estimate line for what is left or the rest', () => {
		expect(levelEstimateLine(96 * MIB * 7200, low, 'left')).toBe('about 2 h for what is left');
		expect(levelEstimateLine(96 * MIB * 7200, low, 'rest')).toBe('about 2 h for the rest');
		expect(levelEstimateLine(10, max, 'left')).toBe('as fast as your disk allows');
		expect(levelEstimateLine(0, low, 'left')).toBe('Nothing left to index.');
	});
});

describe('copy helpers', () => {
	it('pluralizes file counts and compared files', () => {
		expect(fileCountText(1)).toBe('1 file');
		expect(fileCountText(472)).toBe('472 files');
		expect(comparedFilesText(1)).toBe('1 new file is being compared with files you already have.');
		expect(comparedFilesText(3)).toBe('3 new files are being compared with files you already have.');
	});

	it('words the instantly indexed small files', () => {
		expect(smallFilesText(1)).toBe('1 small file was indexed instantly.');
		expect(smallFilesText(15)).toBe('15 small files were indexed instantly.');
	});

	it('titles the not-started state with count and size', () => {
		expect(notIndexedTitle(digests({ pending_total: 472, pending_bytes: 1.75 * 1024 * GIB }))).toBe(
			'472 files (1.75 TB) are not indexed yet'
		);
		expect(notIndexedTitle(digests({ pending_total: 1, pending_bytes: GIB }))).toBe('1 file (1 GB) is not indexed yet');
	});

	it('summarizes a pick with the time at the current level', () => {
		expect(pickSelectionSummary(0, 0, LEVELS[0])).toBe('Nothing selected yet');
		expect(pickSelectionSummary(3, 96 * MIB * 240, LEVELS[0])).toBe('3 selected · 22.5 GB · about 4 min at Low');
		expect(pickSelectionSummary(3, GIB, LEVELS[2])).toBe('3 selected · 1 GB · as fast as your disk allows');
		expect(pickSelectionSummary(3, GIB, null)).toBe('3 selected · 1 GB');
		expect(pickSelectionSummary(3, GIB, null, true)).toBe('Up to 3 selected · 1 GB');
	});
});

describe('indexingScanHasContent', () => {
	it('is false without a status or while idle', () => {
		expect(indexingScanHasContent(null)).toBe(false);
		expect(indexingScanHasContent(status())).toBe(false);
	});

	it('is true for every running and failing scan state', () => {
		for (const state of ['scanning', 'indexing', 'failed', 'cancelled', 'blocked'] as const) {
			expect(indexingScanHasContent(withDigests({}, { state }))).toBe(true);
		}
	});

	it('is false for a finished scan with digests and nothing to report', () => {
		expect(indexingScanHasContent(withDigests())).toBe(false);
	});

	it('is true for a finished scan without digests, with zero files, and with listed problems', () => {
		expect(indexingScanHasContent(status({ state: 'done', found_on_disk: 5, indexed: 5 }))).toBe(true);
		expect(indexingScanHasContent(withDigests({}, { found_on_disk: 0 }))).toBe(true);
		expect(indexingScanHasContent(withDigests({}, { failed_files_total: 2 }))).toBe(true);
		expect(indexingScanHasContent(withDigests({}, { skipped_duplicates_total: 1 }))).toBe(true);
		expect(indexingScanHasContent(withDigests({}, { restart_pending: true }))).toBe(true);
	});
});

describe('indexingDoneSummary with digests', () => {
	it('drops the up-to-date and new-indexed lines but keeps the zero-found case', () => {
		expect(indexingDoneSummary(withDigests({}, { indexed: 20 }))).toBeNull();
		expect(indexingDoneSummary(withDigests({}, { indexed: 0 }))).toBeNull();
		expect(indexingDoneSummary(withDigests({}, { found_on_disk: 0 }))).toEqual({ kind: 'zero' });
	});
});

describe('isSetupPath', () => {
	it('is true for the setup wizard and anything under it', () => {
		expect(isSetupPath('/setup')).toBe(true);
		expect(isSetupPath('/setup/')).toBe(true);
		expect(isSetupPath('/setup/claim')).toBe(true);
	});

	it('is false for every other page, including look-alike names', () => {
		for (const path of ['/', '/generate', '/admin', '/settings', '/setups', '/setup-guide', '/library/setup']) {
			expect(isSetupPath(path)).toBe(false);
		}
	});
});

describe('setupModelsLine', () => {
	const awaiting = { awaiting_choice: true, files: 472, bytes: GIB, min_files: 20, min_bytes: 64 * GIB };

	it('is empty without digests or without models', () => {
		expect(setupModelsLine(null)).toBe('');
		expect(setupModelsLine(digests())).toBe('');
	});

	it('says the models are usable now and that the choice comes after setup while a choice is awaited', () => {
		expect(setupModelsLine(digests({ ready: 15, pending_total: 472, plan: awaiting }))).toBe(
			"487 models found. They're usable now; you'll choose how to index them after setup."
		);
	});

	it('does not promise a choice once it is answered', () => {
		expect(setupModelsLine(digests({ ready: 15, pending_total: 472 }))).toBe("487 models found. They're usable now.");
	});

	it('uses the singular for one model', () => {
		expect(setupModelsLine(digests({ pending_total: 1, plan: awaiting }))).toBe(
			"1 model found. It's usable now; you'll choose how to index it after setup."
		);
		expect(setupModelsLine(digests({ ready: 1 }))).toBe("1 model found. It's usable now.");
	});
});
