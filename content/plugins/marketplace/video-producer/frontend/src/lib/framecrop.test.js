import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import './svelte-harness.js';
import { DOMAINS, LOCALES, mergeDomains, translate } from '../i18n/core.js';

const here = dirname(fileURLToPath(import.meta.url));
const i18nDir = join(here, '..', 'i18n');
const dictionaries = Object.fromEntries(
	LOCALES.map((locale) => [locale, mergeDomains(Object.fromEntries(DOMAINS.map((domain) => [domain, JSON.parse(readFileSync(join(i18nDir, locale, `${domain}.json`), 'utf-8'))])))])
);
const translator = (locale, missing) => (key, params) => translate(dictionaries, locale, key, params, (k, l) => missing.push(`${l}:${k}`));
const source = (path) => readFileSync(new URL(path, import.meta.url), 'utf8');

const { doc, ui, dispose } = await import('./state.svelte.js');
const { project } = await import('./project.svelte.js');
const { estimateState, noteText, readiness } = await import('./production.svelte.js');
const { t } = await import('../i18n/index.svelte.js');
const WAN = '01KX47WANVIDEO0000000000TV';

after(() => dispose());

function scene() {
	project.id = 'p1';
	ui.me = { engines: [{ id: 'native', kind: 'local', name: 'Local' }], presets: [{ id: WAN, name: 'Wan 2.2', video: true, cloud: false, profiles: ['FL2VA'] }] };
	ui.workspace = 'storyboard';
	ui.selection = { kind: 'clip', id: 's1' };
	ui.dirty = true;
	ui.production.target = 'clip';
	doc.engine = { backend: 'native', preset: WAN, model: null };
	doc.direction.mode = 'FL2VA';
	doc.clips = [{ id: 's1', start: 0, dur: 4, label: 'Opening', stage: 'todo', take: null }];
	doc.prompts = [{ id: 'pr1', shotId: 's1', start: 0, dur: 4, strength: 75, text: 'A walk.' }];
}

const cropped = (code, extra = {}) => ({ code, data: { shot_id: 's1', roles: ['first'], size: '1024x768', aspect: '16:9', loss: 23, chained: false, ...extra } });

test('a start image cropped to the frame of the take is a warning that says the share it loses, and a large crop also offers a start image of the shot', () => {
	scene();
	estimateState.data = { blocks: [], warnings: [cropped('frame_cropped'), cropped('frame_cropped_large', { size: '1024x1024', loss: 42 }), { code: 'time_unknown' }] };
	const ready = readiness();
	assert.deepEqual(ready.missing, []);
	assert.deepEqual(ready.warnings, [
		t('generation.warn.frameCropped', { shot: 'Opening', size: '1024×768', aspect: '16:9', loss: 23 }),
		t('generation.warn.frameCroppedLarge', { shot: 'Opening', size: '1024×1024', aspect: '16:9', loss: 42 })
	]);
	assert.deepEqual(ready.notes, [noteText({ code: 'time_unknown' })]);
	assert.deepEqual(ready.crops, ['s1']);
	estimateState.data = { blocks: [], warnings: [cropped('frame_cropped')] };
	assert.deepEqual(readiness().crops, []);
	estimateState.data = null;
});

test('the crop of an end image and of the last image of the previous shot name that image', () => {
	scene();
	assert.equal(noteText(cropped('frame_cropped', { roles: ['last'] })), t('generation.warn.frameCroppedLast', { shot: 'Opening', size: '1024×768', aspect: '16:9', loss: 23 }));
	assert.equal(noteText(cropped('frame_cropped_large', { roles: ['last'] })), t('generation.warn.frameCroppedLastLarge', { shot: 'Opening', size: '1024×768', aspect: '16:9', loss: 23 }));
	assert.equal(noteText(cropped('frame_cropped', { roles: ['first', 'last'] })), t('generation.warn.frameCropped', { shot: 'Opening', size: '1024×768', aspect: '16:9', loss: 23 }));
	assert.equal(noteText(cropped('frame_cropped', { chained: true })), t('generation.warn.frameCroppedChained', { shot: 'Opening', size: '1024×768', aspect: '16:9', loss: 23 }));
	assert.equal(noteText(cropped('frame_cropped_large', { chained: true })), t('generation.warn.frameCroppedChainedLarge', { shot: 'Opening', size: '1024×768', aspect: '16:9', loss: 23 }));
	assert.equal(noteText({ code: 'frame_crop_failed', data: { shot_id: 's1' } }), t('generation.block.frameCropFailed', { shot: 'Opening' }));
});

test('the texts of the crop exist in every language the plugin ships, with the same parameters', () => {
	const missing = [];
	const en = translator('en', missing);
	const fr = translator('fr', missing);
	const params = { shot: 'Opening', size: '1024×1024', aspect: '16:9', loss: 42 };
	assert.equal(en('generation.warn.frameCropped', params), 'The start image of “Opening” is 1024×1024, not 16:9: it is cropped at its centre, 42 % of the picture left out.');
	assert.equal(en('generation.warn.frameCroppedLarge', params), 'The start image of “Opening” is 1024×1024, not 16:9: cropped at its centre, it loses 42 % of the picture. Make a start image in the frame of the film.');
	assert.equal(fr('generation.warn.frameCropped', params), 'L’image de début de « Opening » est en 1024×1024, pas en 16:9 : elle est recadrée au centre, 42 % de l’image en moins.');
	assert.equal(fr('generation.warn.frameCroppedLarge', params), 'L’image de début de « Opening » est en 1024×1024, pas en 16:9 : recadrée au centre, elle perd 42 % de l’image. Faites une image de début au cadre du film.');
	for (const key of ['frameCropped', 'frameCroppedLast', 'frameCroppedChained', 'frameCroppedLarge', 'frameCroppedLastLarge', 'frameCroppedChainedLarge']) {
		for (const locale of LOCALES) assert.match(dictionaries[locale][`generation.warn.${key}`], /\{shot\}.*\{size\}.*\{aspect\}.*\{loss\}/, `${locale} ${key}`);
	}
	for (const locale of LOCALES) {
		assert.ok(dictionaries[locale]['production.deck.startImage'], locale);
		assert.match(dictionaries[locale]['production.deck.startImageTip'], /\{shot\}/, locale);
		assert.match(dictionaries[locale]['generation.block.frameCropFailed'], /\{shot\}/, locale);
	}
	assert.equal(en('keyframes.note.aspect_mismatch', { size: '1024×1024', aspect: '16:9' }), 'The images will be 1024×1024, not 16:9: a take crops them at their centre. Put a picture in the frame of the project first, or choose a model with an output size.');
	assert.equal(fr('keyframes.note.aspect_mismatch', { size: '1024×1024', aspect: '16:9' }), 'Les images seront en 1024×1024, pas en 16:9 : une prise les recadre au centre. Mettez d’abord une image au cadre du projet, ou choisissez un modèle qui a une taille de sortie.');
	assert.deepEqual(missing, []);
});

test('the deck offers to make a start image for the first shot whose image a large crop cuts', () => {
	const deck = source('../components/production/ProductionDeck.svelte');
	assert.match(deck, /const cropClip = \$derived\(blocked \? null : \(ready\.crops\.map\(/);
	assert.match(deck, /\{#if !busy && cropClip\}/);
	assert.match(deck, /onclick=\{\(\) => startImage\(cropClip\)\}/);
	assert.match(deck, /t\('production\.deck\.startImage'\)/);
});

const { renderComponent } = await import('./svelte-harness.js');
const { frameBox } = await import('./framing.js');
const PICKER = new URL('../components/shots/MediaPicker.svelte', import.meta.url).href;

test('a start image thumbnail has the ratio of the film, wide films across the width and tall ones on a fixed height', () => {
	assert.equal(frameBox({ aspect: '16:9' }), `aspect-ratio: ${16 / 9}; width: 100%`);
	assert.equal(frameBox({ aspect: '1:1' }), 'aspect-ratio: 1; width: 100%');
	assert.equal(frameBox({ aspect: '9:16' }), `aspect-ratio: ${9 / 16}; height: 7rem; width: auto; max-width: 100%; margin: 0 auto`);
	assert.equal(frameBox({ aspect: '2.39:1' }), `aspect-ratio: ${2.39}; width: 100%`);
	assert.equal(frameBox({}), `aspect-ratio: ${16 / 9}; width: 100%`);
});

test('the picker of a start image shows the centre the take keeps, in the frame of the film, and other pickers stay 16:9', async () => {
	doc.assets = [{ id: 'a1', kind: 'image', name: 'square.png', src: '/api/media/uploads/square.png', width: 1024, height: 1024 }];
	const framed = await renderComponent(PICKER, { title: 'Start image', value: 'a1', onpick: () => {}, frame: frameBox({ aspect: '9:16' }) });
	assert.match(framed, /<img src="\/api\/media\/uploads\/square\.png" alt="" class="object-cover" style="aspect-ratio: 0\.5625; height: 7rem; width: auto; max-width: 100%; margin: 0 auto" data-frame-thumb=""/);
	const plain = await renderComponent(PICKER, { title: 'Reference', value: 'a1', onpick: () => {} });
	assert.match(plain, /class="aspect-video w-full object-cover"/);
	assert.doesNotMatch(plain, /data-frame-thumb/);
	const empty = await renderComponent(PICKER, { title: 'Start image', value: null, onpick: () => {}, frame: frameBox({ aspect: '1:1' }) });
	assert.match(empty, /style="aspect-ratio: 1; width: 100%"/);
	doc.assets = [];
});

test('the inspector and the shot editor give the start and end image thumbnails the frame of the film', () => {
	const inspector = source('../components/inspector/Inspector.svelte');
	assert.match(inspector, /<img src=\{thumb\} alt="" class="object-cover" style=\{frameBox\(doc\)\} data-frame-thumb \/>/);
	assert.match(inspector, /style=\{frameBox\(doc\)\}>\{t\('inspector\.frames\.none'\)\}<\/span>/);
	const editor = source('../components/modals/ShotEditor.svelte');
	assert.match(editor, /title=\{t\('inspector\.frames\.first'\)\} value=\{draft\.first\} frame=\{frameBox\(doc\)\}/);
	assert.match(editor, /title=\{t\('inspector\.frames\.last'\)\} value=\{draft\.last\} frame=\{frameBox\(doc\)\}/);
});
