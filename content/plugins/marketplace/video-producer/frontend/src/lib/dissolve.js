export const DISSOLVE_KEY = 'dissolve';
export const MAX_DISSOLVE = 2;
export const DISSOLVE_CHOICES = [0.4, 0.6, 1, 1.2, 2];

const STEP_SLACK = 1e-4;
const PLACE_SLACK = 0.5;
const positive = (value) => (typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : 0);
const finite = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : 0);

export const framesOf = (seconds, fps) => Math.floor(finite(seconds) * positive(fps) + 0.5);

export const secondsOf = (frames, fps) => (positive(fps) && frames ? +(frames / fps).toFixed(6) : 0);

export function limitOf(before, clip, fps) {
	const rate = positive(fps);
	const asked = positive(clip?.[DISSOLVE_KEY]);
	if (!before || !clip || !rate || !asked) return { frames: 0, by: null };
	const bounds = [
		['asked', Math.floor(asked * rate + STEP_SLACK)],
		['max', Math.floor(MAX_DISSOLVE * rate + STEP_SLACK)],
		['half', Math.floor(framesOf(before.dur, rate) / 2)],
		['half', Math.floor(framesOf(clip.dur, rate) / 2)]
	];
	const [by, frames] = bounds.reduce((low, bound) => (bound[1] < low[1] ? bound : low));
	return { frames: Math.max(frames, 0), by: frames <= 0 ? (by === 'asked' ? 'frames' : 'half') : by === 'asked' && asked * rate - frames > STEP_SLACK ? 'frames' : by === 'asked' ? null : by };
}

export const dissolveFrames = (before, clip, fps) => limitOf(before, clip, fps).frames;

export const dissolveOf = (before, clip, fps) => secondsOf(dissolveFrames(before, clip, fps), fps);

export function joinedFrames(before, clip, fps) {
	const frames = dissolveFrames(before, clip, fps);
	if (!frames) return 0;
	const overlap = (finite(before.start) + finite(before.dur) - finite(clip.start)) * positive(fps);
	return Math.abs(overlap - frames) <= PLACE_SLACK ? frames : 0;
}

export const joins = (clips, fps) => (clips ?? []).map((c, i) => (i ? joinedFrames(clips[i - 1], c, fps) : 0));

export const dissolves = (clips, fps) => joins(clips, fps).map((frames) => secondsOf(frames, fps));

export function layout(clips, fps, snap) {
	let at = 0;
	let shift = 0;
	return clips.map((clip, i) => {
		if (i) shift += dissolveOf(clips[i - 1], clip, fps);
		const start = shift ? +(snap(at) - shift).toFixed(6) : snap(at);
		at += clip.dur;
		return start;
	});
}

export const dissolveBefore = (clips, i, fps) => (i > 0 ? secondsOf(joinedFrames(clips[i - 1], clips[i], fps), fps) : 0);

export const dissolveAfter = (clips, i, fps) => (i >= 0 && i + 1 < (clips?.length ?? 0) ? secondsOf(joinedFrames(clips[i], clips[i + 1], fps), fps) : 0);

export const lostSeconds = (clips, fps) => +dissolves(clips, fps)
	.reduce((sum, d) => sum + d, 0)
	.toFixed(6);

export function asked(clip) {
	return positive(clip?.[DISSOLVE_KEY]);
}

export function choicesFor(fps) {
	if (!positive(fps)) return DISSOLVE_CHOICES;
	const most = Math.floor(MAX_DISSOLVE * fps + STEP_SLACK);
	return [...new Set(DISSOLVE_CHOICES.map((v) => secondsOf(Math.min(Math.max(1, framesOf(v, fps)), most), fps)))];
}

export function setDissolve(clips, i, value, fps) {
	const clip = clips[i];
	if (!clip || i <= 0) return false;
	const length = positive(value);
	if (!length) {
		delete clip[DISSOLVE_KEY];
		return true;
	}
	clip[DISSOLVE_KEY] = Math.min(length, MAX_DISSOLVE);
	if (dissolveFrames(clips[i - 1], clip, fps) > 0) {
		clips[i - 1].fadeOut = 0;
		clip.fadeIn = 0;
	}
	return true;
}

export function unfaded(clips, i, fps) {
	const clip = clips[i];
	const head = dissolveBefore(clips, i, fps) > 0;
	const tail = dissolveAfter(clips, i, fps) > 0;
	return head || tail ? { ...clip, ...(head ? { fadeIn: 0 } : {}), ...(tail ? { fadeOut: 0 } : {}) } : clip;
}

export function shownSpan(clips, i, fps) {
	const clip = clips[i];
	const head = dissolveBefore(clips, i, fps) / 2;
	const tail = dissolveAfter(clips, i, fps) / 2;
	return { start: clip.start + head, dur: Math.max(clip.dur - head - tail, 0), head, tail };
}

export function indexAt(clips, time, fps) {
	return (clips ?? []).findIndex((clip, i) => {
		const span = shownSpan(clips, i, fps);
		return time >= span.start && time < span.start + span.dur;
	});
}

export function dissolveMarks(clips, fps) {
	const marks = [];
	(clips ?? []).forEach((clip, i) => {
		const d = dissolveBefore(clips, i, fps);
		if (d > 0) marks.push({ id: clip.id, from: clips[i - 1], to: clip, start: clip.start, dur: d });
	});
	return marks;
}

export function mixAt(clips, time, fps) {
	for (let i = 1; i < (clips?.length ?? 0); i++) {
		const d = dissolveBefore(clips, i, fps);
		const clip = clips[i];
		if (d > 0 && time >= clip.start && time < clip.start + d) return { from: clips[i - 1], to: clip, index: i, progress: (time - clip.start) / d };
	}
	return null;
}

export function dissolveGain(clips, i, time, fps) {
	const clip = clips[i];
	if (!clip) return 1;
	let gain = 1;
	const head = dissolveBefore(clips, i, fps);
	if (head > 0 && time < clip.start + head) gain *= Math.sin((Math.max(time - clip.start, 0) / head) * (Math.PI / 2));
	const tail = dissolveAfter(clips, i, fps);
	const next = clips[i + 1];
	if (tail > 0 && time >= next.start) gain *= Math.cos((Math.min(time - next.start, tail) / tail) * (Math.PI / 2));
	return Math.min(Math.max(gain, 0), 1);
}

export function cuts(clips, fps, frames) {
	const overlaps = joins(clips, fps);
	if (!overlaps.some(Boolean)) return clips.map(() => null);
	const edits = clips.map((clip, i) => frames(clip.start) + Math.floor(overlaps[i] / 2));
	return clips.map((clip, i) => {
		const following = i + 1 < clips.length && overlaps[i + 1] > 0;
		if (!overlaps[i] && !following) return null;
		const start = overlaps[i] ? edits[i] : frames(clip.start);
		const end = following ? edits[i + 1] : frames(clip.start + clip.dur);
		return { start, end, overlap: overlaps[i], head: start - frames(clip.start) };
	});
}
