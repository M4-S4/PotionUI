import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { clean, renderComponent } from './svelte-harness.js';
import { setDissolve } from './dissolve.js';

const { doc, ui, act, relayout, select, bladeAt, duplicateSelection, undo, seqDuration, dispose } = await import('./state.svelte.js');
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

test('dissolve flow: the cut window of the Storyboard offers the dissolve, the review of Director AI does not', async () => {
	project();
	const PANEL = new URL('../components/storyboard/SeamPanel.svelte', import.meta.url).href;
	ui.workspace = 'storyboard';
	assert.match(clean(await renderComponent(PANEL, { clip: doc.clips[1], title: false, compare: false })), /data-seam-transition="b"/);
	ui.workspace = 'director';
	assert.doesNotMatch(clean(await renderComponent(PANEL, { clip: doc.clips[1], title: false, compare: false })), /data-seam-transition/);
	ui.workspace = 'media';
});
