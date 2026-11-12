import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DOMAINS, mergeDomains, translate } from '../i18n/core.js';
import { MAX_DISSOLVE, choicesFor, cuts, dissolveGain, dissolveMarks, dissolves, framesOf, indexAt, joins, layout, limitOf, lostSeconds, mixAt, setDissolve, shownSpan, unfaded } from './dissolve.js';
import { MAKERS } from './exchange.js';
import { ghostLayout } from './band.js';
import { shotCuts } from './tracks.js';

const here = dirname(fileURLToPath(import.meta.url));
const tests = join(here, '..', '..', '..', 'tests', 'fixtures');
const cases = JSON.parse(readFileSync(join(tests, 'dissolve_cases.json'), 'utf-8'));
const readDomain = (domain) => JSON.parse(readFileSync(join(here, '..', 'i18n', 'en', `${domain}.json`), 'utf-8'));
const dictionaries = { en: mergeDomains(Object.fromEntries(DOMAINS.map((domain) => [domain, readDomain(domain)]))) };
const close = (actual, expected) => assert.ok(actual.length === expected.length && actual.every((v, i) => Math.abs(v - expected[i]) < 1e-9), `${actual} != ${expected}`);
const snap = (v) => +(Math.round(v / 0.04) * 0.04).toFixed(3);

function laidOut(durations, asks, fps = 25) {
	const clips = durations.map((dur, i) => ({ id: `c${i}`, dur, fadeIn: 0.4, fadeOut: 0.4, ...(asks[i] ? { dissolve: asks[i] } : {}) }));
	layout(clips, fps, snap).forEach((start, i) => (clips[i].start = start));
	return clips;
}

for (const c of cases) {
	test(`dissolve: the shared case "${c.name}" lays out and plays the frames the server does`, () => {
		const clips = c.clips.map((clip, i) => ({ id: `c${i}`, ...clip }));
		close(layout(clips, c.fps, snap), c.starts);
		clips.forEach((clip, i) => (clip.start = c.starts[i]));
		assert.deepEqual(joins(clips, c.fps), c.frames);
		close(dissolves(clips, c.fps), c.applied);
	});
}

test('dissolve: every choice of the page is a whole number of frames at the rate of the project, within the bound', () => {
	assert.deepEqual(choicesFor(25), [0.4, 0.6, 1, 1.2, 2]);
	assert.deepEqual(choicesFor(24), [0.416667, 0.583333, 1, 1.208333, 2]);
	assert.deepEqual(choicesFor(30), [0.4, 0.6, 1, 1.2, 2]);
	for (const fps of [24, 25, 30, 23.976])
		for (const v of choicesFor(fps)) {
			assert.ok(v > 0 && v <= MAX_DISSOLVE, `${v} at ${fps}`);
			const clips = laidOut([10, 10], [0, v], fps);
			assert.equal(limitOf(clips[0], clips[1], fps).by, null, `${v} at ${fps}`);
			assert.equal(joins(clips, fps)[1], framesOf(v, fps), `${v} at ${fps}`);
			assert.ok(framesOf(v, fps) <= Math.floor(MAX_DISSOLVE * fps), `${v} at ${fps}`);
		}
});

test('dissolve: the page says why a dissolve is shorter than asked: half of a short shot, or whole frames', () => {
	assert.deepEqual(limitOf({ dur: 4 }, { dur: 1, dissolve: 1 }, 25), { frames: 12, by: 'half' });
	assert.deepEqual(limitOf({ dur: 4 }, { dur: 4, dissolve: 0.4 }, 24), { frames: 9, by: 'frames' });
	assert.deepEqual(limitOf({ dur: 4 }, { dur: 4, dissolve: 1 }, 25), { frames: 25, by: null });
	assert.deepEqual(limitOf({ dur: 0.04 }, { dur: 4, dissolve: 1 }, 25), { frames: 0, by: 'half' });
});

test('dissolve: choosing one clears the fade to black on both sides of its cut, only when the cut really dissolves', () => {
	const clips = laidOut([4, 4, 4], [0, 0, 0]);
	assert.equal(setDissolve(clips, 0, 1, 25), false);
	assert.equal(setDissolve(clips, 1, 1, 25), true);
	assert.deepEqual([clips[0].fadeIn, clips[0].fadeOut, clips[1].fadeIn, clips[1].fadeOut, clips[1].dissolve], [0.4, 0, 0, 0.4, 1]);
	setDissolve(clips, 1, 9, 25);
	assert.equal(clips[1].dissolve, MAX_DISSOLVE);
	setDissolve(clips, 1, 0, 25);
	assert.ok(!('dissolve' in clips[1]));
	const short = laidOut([0.04, 0.04], [0, 0]);
	setDissolve(short, 1, 1, 25);
	assert.deepEqual([short[1].dissolve, short[0].fadeOut, short[1].fadeIn], [1, 0.4, 0.4]);
});

test('dissolve: the fades of a dissolved cut are not played, the others are', () => {
	const clips = laidOut([4, 4, 4], [0, 1, 0]);
	assert.deepEqual([unfaded(clips, 0, 25).fadeIn, unfaded(clips, 0, 25).fadeOut], [0.4, 0]);
	assert.deepEqual([unfaded(clips, 1, 25).fadeIn, unfaded(clips, 1, 25).fadeOut], [0, 0.4]);
	assert.equal(unfaded(clips, 2, 25), clips[2]);
});

test('dissolve: the timeline draws each shot up to the middle of its dissolves and one mark per dissolve', () => {
	const clips = laidOut([4, 4, 4], [0, 1, 0.6]);
	assert.deepEqual(shownSpan(clips, 0, 25), { start: 0, dur: 3.5, head: 0, tail: 0.5 });
	assert.deepEqual(shownSpan(clips, 1, 25), { start: 3.5, dur: 3.2, head: 0.5, tail: 0.3 });
	assert.deepEqual(shownSpan(clips, 2, 25), { start: 6.7, dur: 3.7, head: 0.3, tail: 0 });
	assert.deepEqual(
		dissolveMarks(clips, 25).map((m) => [m.id, m.from.id, m.start, m.dur]),
		[
			['c1', 'c0', 3, 1],
			['c2', 'c1', 6.4, 0.6]
		]
	);
	assert.equal(lostSeconds(clips, 25), 1.6);
	assert.deepEqual(shownSpan(laidOut([4, 3], [0, 0]), 1, 25), { start: 4, dur: 3, head: 0, tail: 0 });
});

test('dissolve: the shot at a time is the one the timeline draws there, the incoming one from the middle of the dissolve', () => {
	const clips = laidOut([4, 4, 4], [0, 1, 0]);
	assert.deepEqual([0, 3.2, 3.49, 3.5, 3.9, 6.99, 7].map((t) => indexAt(clips, t, 25)), [0, 0, 0, 1, 1, 1, 2]);
	assert.equal(indexAt(clips, 10.99, 25), 2);
	assert.equal(indexAt(clips, 11, 25), -1);
});

test('dissolve: the viewer mixes the two pictures across the dissolve and nowhere else', () => {
	const clips = laidOut([2, 2, 1], [0, 1, 0]);
	assert.equal(mixAt(clips, 0.99, 25), null);
	const half = mixAt(clips, 1.5, 25);
	assert.deepEqual([half.from.id, half.to.id, half.index, half.progress], ['c0', 'c1', 1, 0.5]);
	assert.equal(mixAt(clips, 1, 25).progress, 0);
	assert.equal(mixAt(clips, 2, 25), null);
	assert.equal(mixAt(clips, 3.5, 25), null);
});

test('dissolve: the sound of the two shots crosses at equal power over the same span', () => {
	const clips = laidOut([2, 2, 1], [0, 1, 0]);
	for (const time of [1, 1.25, 1.5, 1.75, 1.99]) {
		const out = dissolveGain(clips, 0, time, 25);
		const into = dissolveGain(clips, 1, time, 25);
		assert.ok(Math.abs(out * out + into * into - 1) < 1e-9, `${time}: ${out}, ${into}`);
	}
	assert.equal(dissolveGain(clips, 0, 0.5, 25), 1);
	assert.equal(dissolveGain(clips, 1, 2.5, 25), 1);
	assert.equal(dissolveGain(clips, 2, 3.5, 25), 1);
	assert.ok(dissolveGain(clips, 1, 1.01, 25) < 0.05 && dissolveGain(clips, 0, 1.99, 25) < 0.05);
});

test('dissolve: a document whose starts do not match its dissolve plays it as a cut on the page too', () => {
	const clips = [
		{ id: 'a', start: 0, dur: 4 },
		{ id: 'b', start: 3.8, dur: 4, dissolve: 1 }
	];
	assert.deepEqual(joins(clips, 25), [0, 0]);
	assert.equal(mixAt(clips, 3.9, 25), null);
	assert.deepEqual(dissolveMarks(clips, 25), []);
	assert.deepEqual(shownSpan(clips, 1, 25), { start: 3.8, dur: 4, head: 0, tail: 0 });
});

test('dissolve: the exchange cut sits in the middle of the dissolve, in frames', () => {
	const clips = laidOut([5, 4.2, 3], [0, 1, 0.52]);
	const frames = (s) => Math.round(s * 25);
	assert.deepEqual(cuts(clips, 25, frames), [
		{ start: 0, end: 112, overlap: 0, head: 0 },
		{ start: 112, end: 198, overlap: 25, head: 12 },
		{ start: 198, end: 267, overlap: 13, head: 6 }
	]);
	assert.deepEqual(cuts(laidOut([5, 4], [0, 0]), 25, frames), [null, null]);
});

test('dissolve: a sound sliced on the shots is cut at the middle of each dissolve, and a ghost shot lands at the end of the film', () => {
	const clips = laidOut([4, 4, 4], [0, 1, 0]);
	assert.deepEqual(shotCuts({ start: 0, dur: 11 }, clips, 25), [3.5, 7]);
	assert.deepEqual(shotCuts({ start: 0, dur: 12 }, laidOut([4, 4, 4], [0, 0, 0]), 25), [4, 8]);
	assert.equal(ghostLayout(clips, [{ keep: true, dur: 2 }], 3, lostSeconds(clips, 25)).ghosts[0].start, 11);
	assert.equal(ghostLayout(clips, [{ keep: true, dur: 2 }], 3).ghosts[0].start, 12);
});

test('dissolve: the page writes the EDL, OTIO and FCPXML of a project with dissolves as the server does, byte for byte', () => {
	const fixture = JSON.parse(readFileSync(join(tests, 'exports', 'dissolve-project.json'), 'utf-8'));
	const fallback = translate(dictionaries, 'en', 'lib.default.clip');
	const label = (c) => c?.label || fixture.doc.assets?.find((a) => a.id === c?.assetId)?.name || fallback;
	for (const format of ['edl', 'otio', 'fcpxml']) {
		const golden = readFileSync(join(tests, 'exports', `dissolve.${format}`));
		assert.ok(Buffer.from(MAKERS[format](fixture.doc, { label }), 'utf-8').equals(golden), `${format} differs from dissolve.${format}`);
	}
	const edl = MAKERS.edl(fixture.doc, { label }).split('\r\n');
	assert.ok(edl.includes('002  VP002    V     D    025 00:00:00:10 00:00:02:06 00:00:04:00 00:00:07:17'));
	assert.ok(edl.includes('004  VP004    V     D    037 00:00:00:06 00:00:03:19 00:00:09:05 00:00:12:18'));
});
