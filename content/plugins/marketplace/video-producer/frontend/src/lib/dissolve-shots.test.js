import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import './svelte-harness.js';
import { setDissolve, mixAt } from './dissolve.js';
import { shownShots } from './monitor.js';

globalThis.document ??= { body: {}, querySelector: () => null };

const { doc, ui, act, relayout, select, relink, clipAt, pickClip, entryTime, undo, dispose } = await import('./state.svelte.js');
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

const heard = [];
window.__potionui.notifications.toast = (level, message) => heard.push(message);
const settled = () => new Promise((resolve) => queueMicrotask(resolve));
const { shotsIn, checkpoint, settle, adopt, replaceDoc } = await import('./state.svelte.js');
const { pickTarget } = await import('./production.svelte.js');
const { placement } = await import('./dissolve.js');
const { commandById } = await import('./commands.js');
const { shotTrim } = await import('./trims.js');
const { drag } = await import('./timeline.js');
const { t } = await import('../i18n/index.svelte.js');

test('dissolve shots: a range made from a shot target covers the shot as the timeline draws it, and a range by time keeps the shots it draws', () => {
	scene();
	select('clip', 'b');
	assert.deepEqual(ids(jobClips()), ['b']);
	pickTarget('range');
	assert.deepEqual([ui.range.in, ui.range.out], [3.5, 6.5]);
	assert.deepEqual(ids(jobClips()), ['b']);
	assert.deepEqual(readiness().range, [3.5, 6.5]);
	doc.prompts = [prompt('pa', 'a', 0, ''), prompt('pb', 'b', 3, 'A walk.'), prompt('pc', 'c', 6, '')];
	assert.deepEqual(ids(productionItems().prompts), ['pb']);
	assert.ok(!readiness().reasons.some((r) => r.code === 'emptyPrompts'));
	assert.deepEqual(ids(shotsIn([0, 3.2])), ['a']);
	assert.deepEqual(ids(shotsIn([0, 3.8])), ['a', 'b']);
	assert.deepEqual(ids(shotsIn([6.4, 9])), ['b', 'c']);
	assert.deepEqual(ids(shotsIn([6.6, 9])), ['c']);
	scene([0, 0, 0]);
	select('clip', 'b');
	pickTarget('range');
	assert.deepEqual([ui.range.in, ui.range.out], [4, 8]);
	assert.deepEqual(ids(jobClips()), ['b']);
	assert.deepEqual(ids(shotsIn([0, 4.01])), ['a', 'b']);
});

function seeded(seed) {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let x = a;
		x = Math.imul(x ^ (x >>> 15), x | 1);
		x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
		return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
	};
}

test('dissolve shots: on projects drawn at random, going to a shot gives a time where it plays alone, or the meeting point of the two dissolves that take it whole', () => {
	scene([0, 0, 0]);
	doc.clips = [shot('a', 3.52), shot('b', 2, { dissolve: 0.56 })];
	relayout();
	assert.equal(doc.clips[1].start, 2.96);
	assert.equal(mixAt(doc.clips, entryTime(doc.clips[1])), null);
	assert.equal(clipAt(entryTime(doc.clips[1])).id, 'b');
	const random = seeded(1110);
	const pick = (list) => list[Math.floor(random() * list.length)];
	let alone = 0;
	let met = 0;
	for (let n = 0; n < 400; n++) {
		const count = 2 + Math.floor(random() * 6);
		doc.clips = Array.from({ length: count }, (_, i) => shot(`s${i}`, (4 + Math.floor(random() * 120)) * 0.04, i && random() < 0.7 ? { dissolve: pick([0.08, 0.12, 0.4, 0.6, 1, 1.2, 2]) } : {}));
		relayout();
		const places = placement(doc.clips);
		doc.clips.forEach((c, i) => {
			const time = entryTime(c);
			const place = places[i];
			const next = places[i + 1];
			const until = next?.overlap > 0 ? next.start : place.end;
			if (place.start + place.overlap < until) {
				alone += 1;
				assert.equal(mixAt(doc.clips, time), null, `${c.id} of ${JSON.stringify(doc.clips.map((x) => [x.dur, x.dissolve ?? 0]))}`);
				assert.equal(clipAt(time)?.id, c.id);
			} else {
				met += 1;
				assert.equal(time, place.start + place.overlap);
				ui.playhead = time;
				showShot(c);
				assert.equal(ui.playhead, time);
			}
		});
	}
	assert.ok(alone > 1000 && met > 0, `${alone} alone, ${met} met`);
});

test('dissolve shots: without dissolve, a head trim from the keyboard writes the playhead as the new start, as the base', () => {
	scene([0, 0, 0]);
	doc.clips = [shot('a', 2, { start: 0.04 })];
	select('clip', 'a');
	ui.playhead = 0.11;
	commandById('trim-start').run();
	assert.equal(doc.clips[0].start, 0.11);
});

test('dissolve shots: a mouse trim reads the zoom of the moment at each move, as before', () => {
	scene();
	doc.engine = { backend: 'native', preset: null, model: null };
	ui.snapping = false;
	ui.pxPerSec = 100;
	const fire = (type, x) => window.dispatchEvent(Object.assign(new Event(type), { pointerId: 9, clientX: x, clientY: 0 }));
	drag({ button: 0, pointerId: 9, clientX: 0, clientY: 0, stopPropagation() {}, currentTarget: null }, shotTrim(doc.clips[1], 'end', () => ui.pxPerSec));
	fire('pointermove', 20);
	ui.pxPerSec = 50;
	fire('pointermove', 20);
	fire('pointerup', 20);
	assert.equal(doc.clips[1].dur, 4.4);
	ui.pxPerSec = 34;
	ui.snapping = true;
});

test('dissolve shots: a document replaced, or a server write adopted during a drag, never makes the page blame the drag for a limited dissolve', async () => {
	scene([0, 1, 0]);
	await settled();
	let count = heard.length;
	ui.drag = {};
	checkpoint();
	const next = JSON.parse(JSON.stringify(doc));
	next.clips[1].dur = 1.6;
	adopt(next);
	ui.drag = null;
	settle();
	await settled();
	assert.deepEqual(heard.slice(count), []);
	scene([0, 1, 0]);
	await settled();
	count = heard.length;
	ui.drag = {};
	checkpoint();
	replaceDoc({ ...JSON.parse(JSON.stringify(doc)), clips: doc.clips.map((c, i) => ({ ...JSON.parse(JSON.stringify(c)), dur: i === 1 ? 1.6 : c.dur })) });
	ui.drag = null;
	settle();
	await settled();
	assert.deepEqual(heard.slice(count), []);
});

test('dissolve shots: every dissolve a command limits or stops is said, three or more included', async () => {
	scene([0, 0, 0]);
	doc.clips = ['a', 'b', 'c', 'd', 'e'].map((id, i) => shot(id, 2, i ? { dissolve: 1 } : {}));
	relayout();
	await settled();
	let count = heard.length;
	act(() => ((doc.clips[1].dur = 1.6), (doc.clips[3].dur = 1.6), relayout()));
	await settled();
	assert.deepEqual(heard.slice(count), [t('timeline.dissolve.trimLimitedMany', { list: '0.8 s, 0.8 s, 0.8 s, and 0.8 s' })]);
	undo();
	await settled();
	count = heard.length;
	act(() => ((doc.clips[1].dur = 0.12), (doc.clips[3].dur = 1.6), relayout()));
	await settled();
	assert.deepEqual(heard.slice(count), [`${t('timeline.dissolve.trimStoppedMany', { count: 2 })} ${t('timeline.dissolve.trimLimitedMany', { list: '0.8 s and 0.8 s' })}`]);
});

test('dissolve shots: without dissolve, the recap reads the range by time as the base, even with picked shots', async () => {
	const { clean, renderComponent } = await import('./svelte-harness.js');
	const MODALS = new URL('../components/modals/Modals.svelte', import.meta.url).href;
	scene([0, 0, 0]);
	doc.prompts = [prompt('pa', 'a', 0, 'Alpha walks.'), prompt('pb', 'b', 4, 'Bravo runs.'), prompt('pc', 'c', 8, 'Charlie jumps.')];
	select('clip', 'a');
	ui.picked = ['b', 'c'];
	assert.deepEqual(ids(productionItems().prompts), ['pb', 'pc']);
	ui.modal = { type: 'recap' };
	const plain = clean(await renderComponent(MODALS));
	assert.ok(plain.includes('Alpha walks.') && !plain.includes('Bravo runs.'), plain.slice(0, 200));
	scene([0, 1, 0]);
	doc.prompts = [prompt('pa', 'a', 0, 'Alpha walks.'), prompt('pb', 'b', 3, 'Bravo runs.')];
	select('clip', 'a');
	ui.modal = { type: 'recap' };
	const dissolving = clean(await renderComponent(MODALS));
	assert.ok(dissolving.includes('Alpha walks.') && !dissolving.includes('Bravo runs.'));
	ui.modal = null;
	ui.picked = [];
});
