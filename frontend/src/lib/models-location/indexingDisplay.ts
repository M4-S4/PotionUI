import type {
	IndexingDigests,
	IndexingLevel,
	IndexingLevelProfile,
	IndexingStatus
} from '$lib/services/api/models';
import { formatBytes } from '$lib/utils/format';

const RUNNING_STATES = new Set(['scanning', 'indexing']);
const MIB = 1024 * 1024;
const SMALL_ETA_SECONDS = 45;
const MINUTES_BEFORE_HOURS = 90;

export type IndexingDoneSummary =
	| { kind: 'zero' }
	| { kind: 'up_to_date'; found: number }
	| { kind: 'new_indexed'; indexed: number; alreadyIndexed: number; found: number };

export type DigestPhase =
	| 'none'
	| 'choice'
	| 'empty'
	| 'not_started'
	| 'up_to_date'
	| 'paused'
	| 'waiting'
	| 'running';

export function indexingIsRunning(status: IndexingStatus | null): boolean {
	if (!status) return false;
	return RUNNING_STATES.has(status.state) || status.restart_pending === true;
}

export function indexingDigests(status: IndexingStatus | null): IndexingDigests | null {
	return status?.digests ?? null;
}

export function digestsAreRunning(digests: IndexingDigests | null): boolean {
	return !!digests && digests.queued > 0 && !digests.paused;
}

export function indexingIsActive(status: IndexingStatus | null): boolean {
	if (!status) return false;
	if (indexingIsRunning(status)) return true;
	const digests = status.digests;
	if (!digests) return false;
	return digests.plan.awaiting_choice || digestsAreRunning(digests);
}

export function indexingIsVisible(status: IndexingStatus | null): boolean {
	if (!status) return false;
	return status.state !== 'idle' || (status.skipped_duplicates_total ?? 0) > 0;
}

export function indexingScanHasContent(status: IndexingStatus | null): boolean {
	if (!status || !indexingIsVisible(status)) return false;
	if (status.state === 'done') {
		if (indexingDoneSummary(status)) return true;
	} else if (status.state !== 'idle') {
		return true;
	}
	return (
		status.restart_pending === true ||
		(status.failed_files_total ?? 0) > 0 ||
		(status.skipped_duplicates_total ?? 0) > 0 ||
		(status.duplicates?.length ?? 0) > 0 ||
		(status.conflicts?.length ?? 0) > 0
	);
}

export function indexingPercent(status: IndexingStatus | null): number | null {
	if (!status || status.state !== 'indexing') return null;
	const total = status.total ?? 0;
	if (total <= 0) return null;
	const processed = status.processed ?? 0;
	return Math.max(0, Math.min(100, Math.round((processed / total) * 100)));
}

export function indexingDoneSummary(status: IndexingStatus | null): IndexingDoneSummary | null {
	if (!status || status.state !== 'done') return null;
	const found = status.found_on_disk ?? 0;
	if (found === 0) return { kind: 'zero' };
	if (status.digests) return null;
	const indexed = status.indexed ?? 0;
	if (indexed === 0) return { kind: 'up_to_date', found };
	return { kind: 'new_indexed', indexed, alreadyIndexed: found - indexed, found };
}

export function digestPhase(status: IndexingStatus | null): DigestPhase {
	const digests = indexingDigests(status);
	if (!digests) return 'none';
	if (digests.plan.awaiting_choice) return 'choice';
	if (digests.ready + digests.pending_total === 0) return 'empty';
	if (digests.queued === 0) return digests.pending_total > 0 ? 'not_started' : 'up_to_date';
	if (digests.paused) return 'paused';
	if (digests.pause_reason === 'generation') return 'waiting';
	return 'running';
}

export function digestFilesTotal(digests: IndexingDigests): number {
	return digests.ready + digests.pending_total;
}

export function digestBytesDone(digests: IndexingDigests): number {
	const reading = digests.current.reduce((sum, file) => sum + Math.min(file.done, file.size), 0);
	return Math.min(digestBytesTotal(digests), digests.ready_bytes + reading);
}

export function digestBytesTotal(digests: IndexingDigests): number {
	return digests.ready_bytes + digests.pending_bytes;
}

export function digestPercent(digests: IndexingDigests): number {
	const total = digestBytesTotal(digests);
	if (total <= 0) return digests.pending_total === 0 ? 100 : 0;
	return Math.max(0, Math.min(100, Math.round((digestBytesDone(digests) / total) * 100)));
}

export function formatEta(seconds: number | null | undefined): string {
	if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '';
	if (seconds < SMALL_ETA_SECONDS) return 'under a minute';
	const minutes = Math.round(seconds / 60);
	if (minutes < MINUTES_BEFORE_HOURS) return `${minutes} min`;
	const hours = Math.floor(minutes / 60);
	const rest = minutes % 60;
	return rest ? `${hours} h ${rest} m` : `${hours} h`;
}

export function formatEtaAbout(seconds: number | null | undefined): string {
	const text = formatEta(seconds);
	if (!text) return '';
	return seconds != null && seconds < SMALL_ETA_SECONDS ? text : `about ${text}`;
}

export function formatRate(bytesPerSecond: number | null | undefined): string {
	if (!bytesPerSecond || bytesPerSecond <= 0) return '0 MB/s';
	return `${Math.round(bytesPerSecond / MIB)} MB/s`;
}

export function levelLabel(level: IndexingLevel): string {
	return level === 'low' ? 'Low' : level === 'normal' ? 'Normal' : 'Max';
}

export function levelProfile(digests: IndexingDigests | null, level: IndexingLevel): IndexingLevelProfile | null {
	return digests?.levels.find((profile) => profile.id === level) ?? null;
}

const WORKER_WORDS: Record<number, string> = { 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four' };

function workersPhrase(count: number): string {
	const word = WORKER_WORDS[count] ?? String(count);
	return count === 1 ? `${word} worker` : `${word} workers`;
}

function priorityPhrase(profile: IndexingLevelProfile): string {
	const rate = formatRate(profile.rate_bps);
	if (profile.rate_bps == null) return 'at full speed, limited only by your disk';
	if (profile.id === 'low') return `at the lowest priority, reading at most ${rate}`;
	if (profile.id === 'normal') return `at a gentler-than-usual priority, up to ${rate}`;
	return `at full speed, up to ${rate}`;
}

export function levelDescription(profile: IndexingLevelProfile): string {
	const pause = profile.pauses_for_generation
		? 'Pauses while you generate.'
		: 'Does not pause for generations, so expect everything to feel slower.';
	return `${workersPhrase(profile.workers)} ${priorityPhrase(profile)}. ${pause}`;
}

const CORE_WORDS: Record<number, string> = { 1: 'one core', 2: 'two cores', 3: 'three cores', 4: 'four cores' };

export function levelUsageText(profile: IndexingLevelProfile): string {
	const cores = CORE_WORDS[profile.workers] ?? `${profile.workers} cores`;
	if (profile.rate_bps == null) return `Uses up to ${cores} and as much of your disk as it can.`;
	return `Uses at most ${cores} and ${formatRate(profile.rate_bps)}.`;
}

export function smallFilesText(count: number): string {
	return count === 1 ? '1 small file was indexed instantly.' : `${count} small files were indexed instantly.`;
}

export function levelSeconds(bytes: number, profile: IndexingLevelProfile | null): number | null {
	if (!profile || profile.rate_bps == null || profile.rate_bps <= 0) return null;
	return bytes / profile.rate_bps;
}

export function levelTimeText(bytes: number, profile: IndexingLevelProfile | null): string {
	if (!profile) return '';
	const seconds = levelSeconds(bytes, profile);
	if (seconds == null) return 'as fast as your disk allows';
	return formatEtaAbout(seconds);
}

export function levelEstimateLine(bytes: number, profile: IndexingLevelProfile | null, remainder: 'left' | 'rest'): string {
	if (bytes <= 0) return 'Nothing left to index.';
	const time = levelTimeText(bytes, profile);
	if (!time || levelSeconds(bytes, profile) == null) return time;
	return `${time} for ${remainder === 'left' ? 'what is left' : 'the rest'}`;
}

export function isSetupPath(pathname: string): boolean {
	return pathname === '/setup' || pathname.startsWith('/setup/');
}

export function setupModelsLine(digests: IndexingDigests | null): string {
	if (!digests) return '';
	const total = digests.ready + digests.pending_total;
	if (total <= 0) return '';
	const single = total === 1;
	const usable = `${single ? '1 model found.' : `${total} models found.`} ${single ? "It's" : "They're"} usable now`;
	if (!digests.plan.awaiting_choice) return `${usable}.`;
	return `${usable}; you'll choose how to index ${single ? 'it' : 'them'} after setup.`;
}

export function fileCountText(count: number): string {
	return `${count} ${count === 1 ? 'file' : 'files'}`;
}

export function comparedFilesText(count: number): string {
	return count === 1
		? '1 new file is being compared with files you already have.'
		: `${count} new files are being compared with files you already have.`;
}

export function notIndexedTitle(digests: IndexingDigests): string {
	return `${fileCountText(digests.pending_total)} (${formatBytes(digests.pending_bytes, 2)}) ${
		digests.pending_total === 1 ? 'is' : 'are'
	} not indexed yet`;
}

export function pickSelectionSummary(
	count: number,
	bytes: number,
	profile: IndexingLevelProfile | null,
	approximate = false
): string {
	if (count === 0) return 'Nothing selected yet';
	const head = `${approximate ? 'Up to ' : ''}${count} selected · ${formatBytes(bytes, 1)}`;
	const time = levelTimeText(bytes, profile);
	if (!profile || !time) return head;
	return profile.rate_bps == null ? `${head} · ${time}` : `${head} · ${time} at ${levelLabel(profile.id)}`;
}
