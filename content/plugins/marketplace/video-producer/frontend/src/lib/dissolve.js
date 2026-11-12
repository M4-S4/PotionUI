export const DISSOLVE_KEY = 'dissolve';
export const MAX_DISSOLVE = 2;
export const DISSOLVE_GRID = 0.04;
export const DISSOLVE_CHOICES = [0.4, 0.6, 1, 1.2, 2];

const STEP_SLACK = 1e-6;
const seconds = (value) => (typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : 0);

export function dissolveOf(before, clip) {
	if (!before || !clip) return 0;
	const asked = seconds(clip[DISSOLVE_KEY]);
	if (asked <= 0) return 0;
	const length = Math.min(asked, MAX_DISSOLVE, seconds(before.dur) / 2, seconds(clip.dur) / 2);
	const steps = Math.floor(length / DISSOLVE_GRID + STEP_SLACK);
	return steps >= 1 ? Math.round(steps * DISSOLVE_GRID * 1e6) / 1e6 : 0;
}

export const dissolves = (clips) => (clips ?? []).map((c, i) => (i ? dissolveOf(clips[i - 1], c) : 0));

export const dissolveBefore = (clips, i) => (i > 0 ? dissolveOf(clips[i - 1], clips[i]) : 0);

export const dissolveAfter = (clips, i) => (i >= 0 && i + 1 < (clips?.length ?? 0) ? dissolveOf(clips[i], clips[i + 1]) : 0);

export const lostSeconds = (clips) => Math.round(dissolves(clips).reduce((sum, d) => sum + d, 0) * 1e6) / 1e6;

export function asked(clip) {
	return seconds(clip?.[DISSOLVE_KEY]);
}

export function setDissolve(clips, i, value) {
	const clip = clips[i];
	if (!clip || i <= 0) return false;
	const length = seconds(value);
	if (length > 0) {
		clip[DISSOLVE_KEY] = Math.min(length, MAX_DISSOLVE);
		clips[i - 1].fadeOut = 0;
		clip.fadeIn = 0;
	} else delete clip[DISSOLVE_KEY];
	return true;
}

export function unfaded(clips, i) {
	const clip = clips[i];
	const head = dissolveBefore(clips, i) > 0;
	const tail = dissolveAfter(clips, i) > 0;
	return head || tail ? { ...clip, ...(head ? { fadeIn: 0 } : {}), ...(tail ? { fadeOut: 0 } : {}) } : clip;
}

export function shownSpan(clips, i) {
	const clip = clips[i];
	const head = dissolveBefore(clips, i) / 2;
	const tail = dissolveAfter(clips, i) / 2;
	return { start: clip.start + head, dur: Math.max(clip.dur - head - tail, 0), head, tail };
}

export function dissolveMarks(clips) {
	const marks = [];
	(clips ?? []).forEach((clip, i) => {
		const d = dissolveBefore(clips, i);
		if (d > 0) marks.push({ id: clip.id, from: clips[i - 1], to: clip, start: clip.start, dur: d });
	});
	return marks;
}

export function cuts(clips, frames) {
	const joins = dissolves(clips);
	const overlaps = clips.map((clip, i) => (i && joins[i] > 0 ? Math.max(frames(clips[i - 1].start + clips[i - 1].dur) - frames(clip.start), 0) : 0));
	if (!overlaps.some(Boolean)) return clips.map(() => null);
	const edits = clips.map((clip, i) => frames(clip.start) + Math.floor(overlaps[i] / 2));
	return clips.map((clip, i) => {
		const next = i + 1 < clips.length && overlaps[i + 1] > 0;
		if (!overlaps[i] && !next) return null;
		const start = overlaps[i] ? edits[i] : frames(clip.start);
		const end = next ? edits[i + 1] : frames(clip.start + clip.dur);
		return { start, end, overlap: overlaps[i], head: start - frames(clip.start) };
	});
}

export function mixAt(clips, time) {
	for (let i = 1; i < (clips?.length ?? 0); i++) {
		const d = dissolveBefore(clips, i);
		const clip = clips[i];
		if (d > 0 && time >= clip.start && time < clip.start + d) return { from: clips[i - 1], to: clip, index: i, progress: (time - clip.start) / d };
	}
	return null;
}

export function dissolveGain(clips, i, time) {
	const clip = clips[i];
	if (!clip) return 1;
	let gain = 1;
	const head = dissolveBefore(clips, i);
	if (head > 0 && time < clip.start + head) gain *= Math.sin((Math.max(time - clip.start, 0) / head) * (Math.PI / 2));
	const tail = dissolveAfter(clips, i);
	const next = clips[i + 1];
	if (tail > 0 && time >= next.start) gain *= Math.cos((Math.min(time - next.start, tail) / tail) * (Math.PI / 2));
	return Math.min(Math.max(gain, 0), 1);
}
