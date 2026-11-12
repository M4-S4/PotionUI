import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import './svelte-harness.js';
import { setDissolve, mixAt } from './dissolve.js';
import { shownShots } from './monitor.js';

globalThis.document ??= { body: {}, querySelector: () => null };

const { doc, ui, act, relayout, select, relink, clipAt, pickClip, entryTime, dispose } = await import('./state.svelte.js');
const { readiness, productionItems, jobClips, estimateState } = await import('./production.svelte.js');
const { project } = await import('./project.svelte.js');
const { pickShot, stepShot } = await import('./band.svelte.js');
const { showShot } = await import('./monitor.svelte.js');
const { handleKeydown } = await import('./commands.js');

after(() => dispose());

const WAN = '01KX47WANVIDEO0000000000TV';
const shot = (id, dur, extra = {}) => ({ id, start: 0, dur, srcIn: 0, label: `Shot ${id}`, stage: 'todo', take: null, assetId: `m${id}`, fadeIn: 0, fadeOut: 0, ...extra });
const ids = (list) => list.map((x) => x.id);

function scene(dissolves = [0, 1, 1]) {
	project.id = 'p1';
	ui.me = { engines: [{ id: 'native', kind: 'local', name: 'Local' }], presets: [{ id: WAN, name: 'Wan 2.2', video: true, cloud: false, profiles: ['FL2VA'] }] };
	ui.workspace = 'media';
	ui.storyView = 'grid';
	ui.production.target = 'clip';
	ui.preview = null;
	ui.compare = null;
	ui.seamCompare = null;
	ui.picked = [];
	ui.range = null;
	doc.fps = 25;
	doc.engine = { backend: 'native', preset: WAN, model: null };
	doc.direction = { ...doc.direction, mode: 'FL2VA', univers: '', personnages: '', photographie: '', palette: '', rythme: '' };
	doc.assets = ['a', 'b', 'c'].map((id) => ({ id: `m${id}`, kind: 'video', name: `Take ${id}`, src: `/m/${id}.mp4`, width: 1280, height: 720, srcDur: 8 }));
	doc.clips = [shot('a', 4), shot('b', 4), shot('c', 4)];
	doc.prompts = [];
	doc.frames = [];
	doc.audio = [];
	doc.titles = [];
	doc.markers = [];
	doc.references = [];
	act(() => {
		dissolves.forEach((seconds, i) => seconds && setDissolve(doc.clips, i, seconds));
		relayout();
	});
	estimateState.data = null;
}

const prompt = (id, shotId, start, text) => ({ id, shotId, start, dur: 4, strength: 75, text });

test('dissolve shots: a shot target keeps only its own prompts, frames and references, never those of the shots it dissolves with', () => {
	scene();
	assert.deepEqual(doc.clips.map((c) => c.start), [0, 3, 6]);
	doc.prompts = [prompt('pa', 'a', 0, ''), prompt('pc', 'c', 6, 'The bars slide in.')];
	doc.frames = [{ id: 'fa', shotId: 'a', time: 3.96, assetId: 'ma', role: 'Fin', strength: 75 }];
	select('clip', 'b');
	assert.deepEqual(ids(jobClips()), ['b']);
	assert.deepEqual(ids(productionItems().prompts), []);
	assert.deepEqual(ids(productionItems().frames), []);
	const codes = readiness().reasons.map((r) => r.code);
	assert.ok(codes.includes('noDirection'), codes.join());
	assert.ok(!codes.includes('emptyPrompts'), codes.join());
	doc.prompts.push(prompt('pb', 'b', 3, 'A walk.'));
	assert.deepEqual(ids(productionItems().prompts), ['pb']);
	assert.ok(!readiness().reasons.some((r) => r.code === 'noDirection' || r.code === 'emptyPrompts'));
	pickClip('b', 'replace');
	assert.deepEqual(ids(productionItems().prompts), ['pb']);
	ui.picked = [];
	ui.production.target = 'sequence';
	assert.deepEqual(ids(productionItems().prompts), ['pa', 'pc', 'pb']);
	assert.ok(readiness().reasons.some((r) => r.code === 'emptyPrompts'));
	ui.production.target = 'clip';
	assert.deepEqual(ids(productionItems(undefined, { clips: [doc.clips[2]] }).prompts), ['pc']);
});

test('dissolve shots: in a project without dissolve, a shot target keeps the base reading by time', () => {
	scene([0, 0, 0]);
	doc.clips[0].dur = 4.13;
	relayout();
	doc.prompts = [prompt('pa', 'a', 0, ''), prompt('pb', 'b', 4.12, 'A walk.')];
	select('clip', 'a');
	assert.deepEqual(ids(readiness().items.prompts), ['pa', 'pb']);
});

test('dissolve shots: in a project that dissolves, a last frame dragged into a dissolve stays on the outgoing shot, and a first frame on the incoming one', () => {
	scene([0, 1, 0]);
	assert.deepEqual(doc.clips.map((c) => c.start), [0, 3, 7]);
	const end = { id: 'fa', shotId: 'a', time: 3.96, assetId: 'ma', role: 'Fin', strength: 75 };
	const start = { id: 'fb', shotId: 'b', time: 3.04, assetId: 'mb', role: 'Début', strength: 75 };
	const any = { id: 'ft', shotId: 'a', time: 3.96, assetId: 'ma', role: 'Transition', strength: 75 };
	doc.frames = [end, start, any];
	for (const frame of doc.frames) relink(frame);
	assert.deepEqual(doc.frames.map((f) => [f.id, f.shotId]), [
		['fa', 'a'],
		['fb', 'b'],
		['ft', 'b']
	]);
	doc.frames[0].time = 3.2;
	doc.frames[1].time = 3.9;
	for (const frame of doc.frames) relink(frame);
	assert.deepEqual(doc.frames.map((f) => f.shotId).slice(0, 2), ['a', 'b']);
	doc.frames[0].time = 4.5;
	relink(doc.frames[0]);
	assert.equal(doc.frames[0].shotId, 'b');
	scene([0, 0, 0]);
	doc.frames = [{ id: 'fa', shotId: 'a', time: 4.04, assetId: 'ma', role: 'Fin', strength: 75 }];
	relink(doc.frames[0]);
	assert.equal(doc.frames[0].shotId, 'b');
});

test('dissolve shots: going to a shot in a project that dissolves puts the playhead on its first frame out of the dissolve, where it plays alone', () => {
	const key = (name) => handleKeydown({ key: name, code: name, altKey: false, ctrlKey: false, metaKey: false, shiftKey: false, repeat: false, target: null, preventDefault() {} }, null);
	scene([0, 1, 0]);
	const b = doc.clips[1];
	assert.equal(entryTime(b), 4);
	assert.equal(entryTime(doc.clips[0]), 0.01);
	assert.equal(entryTime(doc.clips[2]), 7.01);
	pickShot(b);
	assert.equal(ui.playhead, 4);
	assert.equal(clipAt(ui.playhead).id, 'b');
	assert.equal(mixAt(doc.clips, ui.playhead), null);
	ui.storyView = 'band';
	ui.workspace = 'storyboard';
	select('clip', 'a');
	stepShot(1);
	assert.equal(ui.playhead, 4);
	ui.storyView = 'grid';
	select('clip', 'a');
	ui.playhead = 0;
	key('ArrowRight');
	assert.deepEqual([ui.selection.id, ui.playhead], ['b', 4]);
	ui.workspace = 'media';
	ui.playhead = 3.6;
	assert.equal(clipAt(3.6).id, 'b');
	showShot(b);
	assert.equal(ui.playhead, 4);
	ui.playhead = 5;
	showShot(b);
	assert.equal(ui.playhead, 5);
	scene([0, 0, 0]);
	pickShot(doc.clips[1]);
	assert.equal(ui.playhead, 4.01);
	ui.playhead = 5;
	showShot(doc.clips[1]);
	assert.equal(ui.playhead, 5);
});

test('dissolve shots: during a dissolve the monitor shows both shots, and outside it the one under the playhead', () => {
	scene([0, 1, 0]);
	const view = (playhead) => ({ clips: doc.clips, playhead });
	const find = { take: () => null, asset: () => null };
	assert.deepEqual(shownShots(view(2.5), find), ['a']);
	assert.deepEqual(shownShots(view(3.2), find), ['a', 'b']);
	assert.deepEqual(shownShots(view(3.8), find), ['a', 'b']);
	assert.deepEqual(shownShots(view(4), find), ['b']);
	scene([0, 0, 0]);
	assert.deepEqual(shownShots(view(3.99), find), ['a']);
	assert.deepEqual(shownShots(view(4), find), ['b']);
});
