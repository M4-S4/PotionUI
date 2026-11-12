import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DOMAINS, mergeDomains, translate } from '../i18n/core.js';
import { DISSOLVE_CHOICES, DISSOLVE_GRID, MAX_DISSOLVE, cuts, dissolveGain, dissolveMarks, dissolves, lostSeconds, mixAt, setDissolve, shownSpan, unfaded } from './dissolve.js';
import { MAKERS } from './exchange.js';

const here = dirname(fileURLToPath(import.meta.url));
const tests = join(here, '..', '..', '..', 'tests', 'fixtures');
const cases = JSON.parse(readFileSync(join(tests, 'dissolve_cases.json'), 'utf-8'));
const readDomain = (domain) => JSON.parse(readFileSync(join(here, '..', 'i18n', 'en', `${domain}.json`), 'utf-8'));
const dictionaries = { en: mergeDomains(Object.fromEntries(DOMAINS.map((domain) => [domain, readDomain(domain)]))) };
const close = (actual, expected) => assert.ok(actual.length === expected.length && actual.every((v, i) => Math.abs(v - expected[i]) < 1e-9), `${actual} != ${expected}`);
const clipsOf = (c) => c.clips.map((clip, i) => ({ id: `c${i}`, ...clip }));

function laidOut(durations, joins) {
	const clips = durations.map((dur, i) => ({ id: `c${i}`, dur, fadeIn: 0.4, fadeOut: 0.4, ...(joins[i] ? { dissolve: joins[i] } : {}) }));
	let at = 0;
	clips.forEach((clip, i) => {
		if (i) at -= dissolves(clips)[i];
		clip.start = Math.round(at * 1e6) / 1e6;
		at += clip.dur;
	});
	return clips;
}

for (const c of cases) {
	test(`dissolve: the shared case "${c.name}" gives the dissolve the server gives`, () => {
		close(dissolves(clipsOf(c)), c.applied);
	});
}

test('dissolve: every choice of the page sits on the timeline grid and within the bound', () => {
	for (const v of DISSOLVE_CHOICES) {
		assert.ok(v > 0 && v <= MAX_DISSOLVE);
		assert.ok(Math.abs(Math.round(v / DISSOLVE_GRID) * DISSOLVE_GRID - v) < 1e-9, `${v} is off the grid`);
		assert.equal(dissolves([{ dur: 10 }, { dur: 10, dissolve: v }])[1], v);
	}
});

test('dissolve: choosing one clears the fade to black on both sides of its cut, choosing a cut removes the field', () => {
	const clips = laidOut([4, 4, 4], [0, 0, 0]);
	assert.equal(setDissolve(clips, 0, 1), false);
	assert.equal(setDissolve(clips, 1, 1), true);
	assert.deepEqual([clips[0].fadeIn, clips[0].fadeOut, clips[1].fadeIn, clips[1].fadeOut, clips[1].dissolve], [0.4, 0, 0, 0.4, 1]);
	setDissolve(clips, 1, 9);
	assert.equal(clips[1].dissolve, MAX_DISSOLVE);
	setDissolve(clips, 1, 0);
	assert.ok(!('dissolve' in clips[1]));
});

test('dissolve: the fades of a dissolved cut are not played, the others are', () => {
	const clips = laidOut([4, 4, 4], [0, 1, 0]);
	assert.deepEqual([unfaded(clips, 0).fadeIn, unfaded(clips, 0).fadeOut], [0.4, 0]);
	assert.deepEqual([unfaded(clips, 1).fadeIn, unfaded(clips, 1).fadeOut], [0, 0.4]);
	assert.equal(unfaded(clips, 2), clips[2]);
});

test('dissolve: the timeline draws each shot up to the middle of its dissolves and one mark per dissolve', () => {
	const clips = laidOut([4, 4, 4], [0, 1, 0.6]);
	assert.deepEqual(shownSpan(clips, 0), { start: 0, dur: 3.5, head: 0, tail: 0.5 });
	assert.deepEqual(shownSpan(clips, 1), { start: 3.5, dur: 3.2, head: 0.5, tail: 0.3 });
	assert.deepEqual(shownSpan(clips, 2), { start: 6.7, dur: 3.7, head: 0.3, tail: 0 });
	assert.deepEqual(
		dissolveMarks(clips).map((m) => [m.id, m.from.id, m.start, m.dur]),
		[
			['c1', 'c0', 3, 1],
			['c2', 'c1', 6.4, 0.6]
		]
	);
	assert.equal(lostSeconds(clips), 1.6);
	assert.deepEqual(shownSpan(laidOut([4, 3], [0, 0]), 1), { start: 4, dur: 3, head: 0, tail: 0 });
});

test('dissolve: the viewer mixes the two pictures across the dissolve and nowhere else', () => {
	const clips = laidOut([2, 2, 1], [0, 1, 0]);
	assert.equal(mixAt(clips, 0.99), null);
	const half = mixAt(clips, 1.5);
	assert.deepEqual([half.from.id, half.to.id, half.index, half.progress], ['c0', 'c1', 1, 0.5]);
	assert.equal(mixAt(clips, 1).progress, 0);
	assert.equal(mixAt(clips, 2), null);
	assert.equal(mixAt(clips, 3.5), null);
});

test('dissolve: the sound of the two shots crosses at equal power over the same span', () => {
	const clips = laidOut([2, 2, 1], [0, 1, 0]);
	for (const time of [1, 1.25, 1.5, 1.75, 1.99]) {
		const out = dissolveGain(clips, 0, time);
		const into = dissolveGain(clips, 1, time);
		assert.ok(Math.abs(out * out + into * into - 1) < 1e-9, `${time}: ${out}, ${into}`);
	}
	assert.equal(dissolveGain(clips, 0, 0.5), 1);
	assert.equal(dissolveGain(clips, 1, 2.5), 1);
	assert.equal(dissolveGain(clips, 2, 3.5), 1);
	assert.ok(dissolveGain(clips, 1, 1.01) < 0.05 && dissolveGain(clips, 0, 1.99) < 0.05);
});

test('dissolve: the exchange cut sits in the middle of the dissolve, in frames', () => {
	const clips = laidOut([5, 4.2, 3], [0, 1, 0.52]);
	const frames = (s) => Math.round(s * 25);
	assert.deepEqual(cuts(clips, frames), [
		{ start: 0, end: 112, overlap: 0, head: 0 },
		{ start: 112, end: 198, overlap: 25, head: 12 },
		{ start: 198, end: 267, overlap: 13, head: 6 }
	]);
	assert.deepEqual(cuts(laidOut([5, 4], [0, 0]), frames), [null, null]);
});

test('dissolve: the page writes the EDL, OTIO and FCPXML of a project with dissolves as the server does, byte for byte', () => {
	const fixture = JSON.parse(readFileSync(join(tests, 'exports', 'dissolve-project.json'), 'utf-8'));
	const fallback = translate(dictionaries, 'en', 'lib.default.clip');
	const label = (c) => c?.label || fixture.doc.assets?.find((a) => a.id === c?.assetId)?.name || fallback;
	for (const format of ['edl', 'otio', 'fcpxml']) {
		const golden = readFileSync(join(tests, 'exports', `dissolve.${format}`));
		assert.ok(Buffer.from(MAKERS[format](fixture.doc, { label }), 'utf-8').equals(golden), `${format} differs from dissolve.${format}`);
	}
});
