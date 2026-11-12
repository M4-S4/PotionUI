import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { clean, renderComponent } from './svelte-harness.js';
import { joins, setDissolve } from './dissolve.js';

const { doc, ui, act, relayout, select, bladeAt, duplicateSelection, undo, seqDuration, clipAt, shotIdAt, dispose } = await import('./state.svelte.js');
const { commandById } = await import('./commands.js');
const { t } = await import('../i18n/index.svelte.js');

const here = dirname(fileURLToPath(import.meta.url));
const cases = JSON.parse(readFileSync(join(here, '..', '..', '..', 'tests', 'fixtures', 'dissolve_cases.json'), 'utf-8'));
const VIEWER = new URL('../components/viewer/Viewer.svelte', import.meta.url).href;
const TIMELINE = new URL('../components/timeline/Timeline.svelte', import.meta.url).href;
const SEAM = new URL('../components/storyboard/SeamTransition.svelte', import.meta.url).href;
const INSPECTOR = new URL('../components/inspector/Inspector.svelte', import.meta.url).href;
const shot = (id, start, dur, extra = {}) => ({ id, start, dur, srcIn: 0, label: `Shot ${id}`, stage: 'draft', take: null, assetId: `m${id}`, fadeIn: 0, fadeOut: 0, ...extra });

after(() => dispose());

function project() {
	doc.fps = 25;
	doc.aspect = '16:9';
	doc.assets = ['a', 'b', 'c'].map((id) => ({ id: `m${id}`, kind: 'video', name: `Take ${id}`, src: `/m/${id}.mp4`, width: 1280, height: 720, srcDur: 6 }));
	doc.clips = [shot('a', 0, 2, { fadeOut: 0.4 }), shot('b', 2, 2, { fadeIn: 0.4, fadeOut: 0.4 }), shot('c', 4, 1, { fadeIn: 0.4 })];
	doc.prompts = [];
	doc.frames = [];
	doc.audio = [];
	doc.titles = [];
	doc.markers = [];
	Object.assign(ui, { workspace: 'media', preview: null, compare: null, playhead: 0, selection: null, picked: [], inspectorOpen: true });
}

function dissolveB(seconds = 1) {
	act(() => {
		setDissolve(doc.clips, 1, seconds);
		relayout();
	});
}

for (const c of cases) {
	test(`dissolve flow: the page lays the shared case "${c.name}" out where the assistant does`, () => {
		doc.clips = c.clips.map((clip, i) => ({ id: `c${i}`, start: 99, srcIn: 0, ...clip }));
		relayout();
		assert.ok(
			doc.clips.every((clip, i) => Math.abs(clip.start - c.starts[i]) < 1e-9),
			doc.clips.map((clip) => clip.start).join(', ')
		);
	});
}

test('dissolve flow: a dissolve shortens the film by its length, moves the later shots, and undoes in one step', () => {
	project();
	assert.equal(seqDuration(), 5);
	dissolveB();
	assert.deepEqual(
		doc.clips.map((c) => [c.start, c.fadeIn, c.fadeOut]),
		[
			[0, 0, 0],
			[1, 0, 0.4],
			[3, 0.4, 0]
		]
	);
	assert.equal(seqDuration(), 4);
	undo();
	assert.deepEqual(
		doc.clips.map((c) => [c.start, c.fadeIn, c.fadeOut, 'dissolve' in c]),
		[
			[0, 0, 0.4, false],
			[2, 0.4, 0.4, false],
			[4, 0.4, 0, false]
		]
	);
});

test('dissolve flow: the blade keeps the dissolve on the head of a shot and a duplicate starts with a cut', () => {
	project();
	dissolveB();
	select('clip', 'b');
	bladeAt(2);
	assert.deepEqual(
		doc.clips.map((c) => [c.dur, c.dissolve ?? 0]),
		[
			[2, 0],
			[1, 1],
			[1, 0],
			[1, 0]
		]
	);
	assert.deepEqual(
		doc.clips.map((c) => c.start),
		[0, 1.52, 2.52, 3.52]
	);
	project();
	dissolveB();
	select('clip', 'b');
	duplicateSelection();
	assert.deepEqual(
		doc.clips.map((c) => c.dissolve ?? 0),
		[0, 1, 0, 0]
	);
	assert.deepEqual(
		doc.clips.map((c) => c.start),
		[0, 1, 3, 5]
	);
});

test('dissolve flow: the viewer lays the incoming picture over the outgoing one at the share of the dissolve gone by', async () => {
	project();
	dissolveB();
	ui.playhead = 1.5;
	const middle = clean(await renderComponent(VIEWER));
	assert.match(middle, /<video[^>]*src="\/m\/a\.mp4"[^>]*style="opacity: 1;/);
	assert.match(middle, /<video[^>]*src="\/m\/b\.mp4"[^>]*data-dissolve-layer="b"[^>]*style="opacity: 0\.5;/);
	ui.playhead = 0.5;
	assert.doesNotMatch(clean(await renderComponent(VIEWER)), /data-dissolve-layer/);
	ui.playhead = 2.5;
	const after = clean(await renderComponent(VIEWER));
	assert.doesNotMatch(after, /data-dissolve-layer/);
	assert.match(after, /<video[^>]*src="\/m\/b\.mp4"/);
});

test('dissolve flow: the timeline marks the dissolve on its cut, draws each shot to the middle of it and hides the fade it replaces', async () => {
	project();
	ui.pxPerSec = 100;
	const before = clean(await renderComponent(TIMELINE));
	assert.doesNotMatch(before, /data-dissolve=/);
	const fadesBefore = (before.match(new RegExp(`aria-label="${t('timeline.fade.in')}"`, 'g')) ?? []).length;
	dissolveB();
	const html = clean(await renderComponent(TIMELINE));
	const tip = t('timeline.dissolve.tip', { dur: '1.0 s' });
	assert.match(html, new RegExp(`aria-label="${tip}"[^>]*data-dissolve="b"`));
	assert.match(html, /data-dissolve="b" style="left: 150px; width: 100px;"/);
	assert.match(html, new RegExp(`aria-label="${t('timeline.clip.label', { name: 'Shot a', stage: t('lib.stage.draft'), dur: '2.0 s' })}" style="left: 0px; width: 150px;`));
	assert.match(html, new RegExp(`aria-label="${t('timeline.clip.label', { name: 'Shot b', stage: t('lib.stage.draft'), dur: '2.0 s' })}" style="left: 150px; width: 150px;`));
	assert.match(html, new RegExp(`aria-label="${t('timeline.clip.label', { name: 'Shot c', stage: t('lib.stage.draft'), dur: '1.0 s' })}" style="left: 300px; width: 100px;`));
	assert.equal((html.match(new RegExp(`aria-label="${t('timeline.fade.in')}"`, 'g')) ?? []).length, fadesBefore - 1);
});

test('dissolve flow: the cut offers the dissolve, says how much shorter the film gets, and the inspector greys the fades it replaces', async () => {
	project();
	const cut = clean(await renderComponent(SEAM, { clip: doc.clips[1] }));
	assert.match(cut, /<option value="0" selected/);
	assert.ok(cut.includes(t('timeline.dissolve.hint')));
	assert.equal(clean(await renderComponent(SEAM, { clip: doc.clips[0] })).trim(), '');
	dissolveB();
	const joined = clean(await renderComponent(SEAM, { clip: doc.clips[1] }));
	assert.match(joined, /<option value="1" selected/);
	assert.ok(joined.includes(t('timeline.dissolve.shorter', { dur: '1.0 s' })));
	select('clip', 'b');
	const inspector = clean(await renderComponent(INSPECTOR));
	assert.ok(inspector.includes(t('inspector.fades.dissolved')));
	assert.match(inspector, /data-seam-transition="b"/);
});

test('dissolve flow: the cut window of the Storyboard grid offers the dissolve, the compact cuts of the band and of the review of Director AI do not', async () => {
	project();
	const PANEL = new URL('../components/storyboard/SeamPanel.svelte', import.meta.url).href;
	assert.match(clean(await renderComponent(PANEL, { clip: doc.clips[1] })), /data-seam-transition="b"/);
	assert.doesNotMatch(clean(await renderComponent(PANEL, { clip: doc.clips[1], title: false, compare: false })), /data-seam-transition/);
});

test('dissolve flow: from the middle of a dissolve on, the shot under the playhead is the incoming one, and the blade refuses to cut inside it', () => {
	project();
	dissolveB();
	assert.deepEqual([0.9, 1.2, 1.49, 1.5, 1.8, 2.99, 3].map((time) => clipAt(time)?.id), ['a', 'a', 'a', 'b', 'b', 'b', 'c']);
	assert.equal(shotIdAt(1.8), 'b');
	select('clip', 'b');
	const before = JSON.stringify(doc.clips);
	bladeAt(1.8);
	assert.equal(JSON.stringify(doc.clips), before);
	bladeAt(2.6);
	assert.deepEqual(
		doc.clips.map((c) => [c.id === 'b' ? 'b' : c.id === 'a' || c.id === 'c' ? c.id : 'new', c.dur]),
		[
			['a', 2],
			['b', 1.6],
			['new', 0.4],
			['c', 1]
		]
	);
});

test('dissolve flow: trimming a dissolved shot from the keyboard keeps its dissolve where it is drawn', () => {
	project();
	dissolveB(0.4);
	select('clip', 'b');
	ui.playhead = 3.2;
	commandById('trim-end').run();
	const expected = [
		[0, 2],
		[1.6, 1.6],
		[3.2, 1]
	];
	assert.ok(
		doc.clips.every((c, i) => Math.abs(c.start - expected[i][0]) < 1e-9 && Math.abs(c.dur - expected[i][1]) < 1e-9),
		JSON.stringify(doc.clips.map((c) => [c.start, c.dur]))
	);
	assert.ok(Math.abs(seqDuration() - 4.2) < 1e-9);
});

test('dissolve flow: an incoming shot without media fades the picture to black, as the export does', async () => {
	project();
	doc.clips[1].assetId = null;
	dissolveB();
	ui.playhead = 1.5;
	const html = clean(await renderComponent(VIEWER));
	assert.match(html, /<div class="pointer-events-none absolute inset-0 bg-black" data-dissolve-layer="b" style="opacity: 0\.5;"><\/div>/);
});

test('dissolve flow: the cut says when a dissolve is rounded to the grid, when it is capped, and when the shot is chained', async () => {
	project();
	doc.fps = 24;
	act(() => {
		setDissolve(doc.clips, 1, 0.5);
		relayout();
	});
	const rounded = clean(await renderComponent(SEAM, { clip: doc.clips[1] }));
	assert.ok(rounded.includes(t('timeline.dissolve.rounded', { dur: '0.48 s' })));
	assert.doesNotMatch(rounded, /data-dissolve-capped/);
	assert.match(rounded, /<option value="0\.4">/);
	assert.match(rounded, /<option value="0\.5" selected/);
	project();
	doc.clips[1].dur = 1;
	doc.clips[2].stage = 'source';
	act(() => {
		setDissolve(doc.clips, 2, 2);
		relayout();
	});
	const capped = clean(await renderComponent(SEAM, { clip: doc.clips[2] }));
	assert.ok(capped.includes(t('timeline.dissolve.capped', { dur: '0.48 s' })));
	assert.doesNotMatch(capped, /data-dissolve-rounded|data-dissolve-chained/);
	doc.clips[2].stage = 'todo';
	assert.ok(clean(await renderComponent(SEAM, { clip: doc.clips[2] })).includes(t('timeline.dissolve.chained')));
	doc.firstFromPrevious = false;
	assert.doesNotMatch(clean(await renderComponent(SEAM, { clip: doc.clips[2] })), /data-dissolve-chained/);
	delete doc.firstFromPrevious;
});

test('dissolve flow: changing the cadence of the project keeps every dissolve and every start', async () => {
	project();
	doc.clips[1].dur = 3.1;
	act(() => {
		setDissolve(doc.clips, 1, 1);
		setDissolve(doc.clips, 2, 0.6);
		relayout();
	});
	const starts = doc.clips.map((c) => c.start);
	const played = joins(doc.clips);
	assert.deepEqual(starts, [0, 1, 3.6]);
	assert.deepEqual(played, [0, 1, 0.5]);
	for (const fps of [24, 25, 30, 48, 50, 60]) {
		doc.fps = fps;
		assert.deepEqual(joins(doc.clips), played, `${fps} fps`);
		relayout();
		assert.deepEqual(
			doc.clips.map((c) => c.start),
			starts,
			`${fps} fps`
		);
		ui.playhead = 1.5;
		assert.match(clean(await renderComponent(VIEWER)), /data-dissolve-layer="b"[^>]*style="opacity: 0\.5;/, `${fps} fps`);
	}
});

test('dissolve flow: at 24 and 48 fps the prompts and frames linked to a shot stay on its start', () => {
	for (const fps of [24, 48]) {
		project();
		doc.fps = fps;
		doc.clips.push(shot('d', 5, 2));
		doc.prompts = doc.clips.map((c, i) => ({ id: `p${i}`, shotId: c.id, start: c.start, dur: c.dur, strength: 75, text: 'x' }));
		doc.frames = doc.clips.map((c, i) => ({ id: `f${i}`, shotId: c.id, time: c.start, assetId: null, role: 'First frame', strength: 75 }));
		act(() => {
			setDissolve(doc.clips, 1, 0.4);
			setDissolve(doc.clips, 2, 0.4);
			setDissolve(doc.clips, 3, 0.4);
			relayout();
		});
		const starts = doc.clips.map((c) => c.start);
		assert.deepEqual(starts, [0, 1.6, 3.2, 3.8], `${fps} fps`);
		assert.deepEqual(
			doc.prompts.map((p) => p.start),
			starts,
			`${fps} fps`
		);
		assert.deepEqual(
			doc.frames.map((f) => f.time),
			starts,
			`${fps} fps`
		);
		assert.deepEqual(
			starts.map((time) => shotIdAt(time)),
			['a', 'b', 'c', 'd'],
			`${fps} fps`
		);
	}
});

test('dissolve flow: trimming the head of a dissolved shot from the keyboard keeps its linked prompt inside it', () => {
	project();
	dissolveB(0.4);
	doc.prompts = [{ id: 'p', shotId: 'b', start: 1.6, dur: 2, strength: 75, text: 'x' }];
	select('clip', 'b');
	ui.playhead = 2;
	commandById('trim-start').run();
	assert.deepEqual(
		doc.clips.map((c) => [c.start, c.dur]),
		[
			[0, 2],
			[1.6, 1.6],
			[3.2, 1]
		]
	);
	assert.equal(doc.prompts[0].start, 1.6);
	assert.equal(shotIdAt(doc.prompts[0].start), 'b');
});

test('dissolve flow: a project whose cuts are all too short for a dissolve trims from the keyboard as before', () => {
	project();
	doc.clips[1].dissolve = 0.03;
	select('clip', 'b');
	ui.playhead = 2.5;
	commandById('trim-start').run();
	assert.deepEqual(
		doc.clips.map((c) => [c.start, c.dur]),
		[
			[0, 2],
			[2.5, 1.5],
			[4, 1]
		]
	);
});
