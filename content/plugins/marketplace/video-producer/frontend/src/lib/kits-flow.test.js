import { test, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const hooks = `
import { readFile } from 'node:fs/promises';
import { compileModule } from ${JSON.stringify(import.meta.resolve('svelte/compiler'))};
export async function load(url, context, next) {
	if (url.endsWith('.svelte.js')) return { format: 'module', shortCircuit: true, source: compileModule(await readFile(new URL(url), 'utf8'), { generate: 'client', filename: url }).js.code };
	if (url.endsWith('.json')) return { format: 'module', shortCircuit: true, source: 'export default ' + (await readFile(new URL(url), 'utf8')) };
	return next(url, context);
}`;
register(`data:text/javascript,${encodeURIComponent(hooks)}`, import.meta.url);

class Media {
	removeAttribute() {}
	load() {}
	remove() {}
	set src(value) {
		queueMicrotask(() => this.onerror?.());
	}
}

const toasts = [];
const mounted = [];
const components = {};
function modal(name) {
	return {
		mount(target, props) {
			const handle = { name, props, open: true };
			mounted.push(handle);
			return handle;
		},
		update(handle, next) {
			Object.assign(handle.props, next);
			if (next.isOpen === false) handle.open = false;
		},
		unmount() {}
	};
}

globalThis.window = { __potionui: { confirm: async () => true, notifications: { toast: (level, message) => toasts.push([level, message]) }, components } };
globalThis.document = { body: { appendChild: (node) => node }, createElement: () => new Media(), querySelector: () => null };
globalThis.Image = Media;

const { doc, ui, replaceDoc, dispose } = await import('./state.svelte.js');
const { bindClient } = await import('./project.svelte.js');
const importer = await import('./importer.js');
const { emptyDoc, newAsset } = await import('./starter.js');
const { setLocale } = await import('../i18n/index.svelte.js');
const { mediaOf, seededKits } = await import('../../../tests/fixtures/make_kit_picks.mjs');

const fixture = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', 'tests', 'fixtures', 'kit_picks.json'), 'utf-8'));

after(() => {
	dispose();
	delete globalThis.window;
	delete globalThis.document;
	delete globalThis.Image;
});

const resolved = [];
bindClient({
	async resolveMedia(origin) {
		resolved.push(origin);
		if (origin.kind !== 'library' || !origin.itemId.startsWith('up-')) return { ok: false, error: { status: 404 } };
		return { ok: true, data: mediaOf(origin) };
	}
});

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

function item(extra = {}) {
	return { id: 'it_face', kit_id: 'kt_mara', kind: 'image', name: 'face', position: 0, media_ref: 'up-image-1', text: null, url: '/api/media/uploads/hands.png', thumbnail_url: null, width: 512, height: 512, duration_seconds: null, missing: false, created_at: '', ...extra };
}

const MARA = {
	id: 'kt_mara',
	name: 'Mara',
	handle: 'mara',
	items: [
		item(),
		item({ id: 'it_bio', kind: 'text', name: 'bio', position: 1, media_ref: null, url: null, text: 'Mara is tall.' }),
		item({ id: 'it_coat', name: 'coat', position: 2, media_ref: 'up-image-3' }),
		item({ id: 'it_walk', kind: 'video', name: 'walk', position: 3, media_ref: 'up-video-2' }),
		item({ id: 'it_gone', name: 'gone', position: 4, media_ref: null, url: null, missing: true }),
		item({ id: 'it_voice', kind: 'audio', name: 'voice', position: 5, media_ref: 'up-audio-5' })
	]
};

const picked = (kitItem, kit = MARA) => ({ key: `kit:${kitItem.id}`, mediaType: kitItem.kind, filename: kitItem.name, origin: { kind: 'kit', kitId: kit.id, itemId: kitItem.id }, kit: { id: kit.id, name: kit.name, handle: kit.handle }, kitItem });
const byId = (id) => MARA.items.find((it) => it.id === id);
const last = () => mounted.at(-1);
const assetsOf = () => doc.assets.map((a) => [a.kind, a.origin, a.name]);
const lib = (ref) => ({ kind: 'library', itemId: ref });

function close(handle) {
	handle.props.onClose();
}

beforeEach(() => {
	setLocale('en');
	replaceDoc(emptyDoc((key) => key));
	ui.hostModal = false;
	toasts.length = 0;
	mounted.length = 0;
	resolved.length = 0;
	for (const key of Object.keys(components)) delete components[key];
	components.MediaPickerModal = modal('MediaPickerModal');
	components.UploadLibraryModal = modal('UploadLibraryModal');
});

test('the kit import opens the host picker on the Kits source only, multiple, for the kinds of the place, with the whole-kit choice and no kits.accepted', () => {
	importer.openKitImport();
	const { name, props } = last();
	assert.equal(name, 'MediaPickerModal');
	assert.deepEqual(props.sources, ['kits']);
	assert.equal(props.initialSource, 'kits');
	assert.deepEqual(props.mediaTypes, ['image', 'video']);
	assert.deepEqual(props.selectableMediaTypes, ['image', 'video']);
	assert.equal(props.multiple, true);
	assert.equal(typeof props.kits.onLoad, 'function');
	assert.equal('accepted' in props.kits, false);
	assert.equal(props.title, 'Take images and videos from a kit');
	assert.equal(ui.hostModal, true);
	close(last());
	assert.equal(ui.hostModal, false);
	importer.openKitImport('audio');
	assert.deepEqual(last().props.mediaTypes, ['audio']);
	assert.equal(last().props.title, 'Take audio from a kit');
});

test('several kit items picked: each becomes an ordinary library asset of its media_ref, in the order picked, and the document gains no other key', async () => {
	const before = Object.keys(JSON.parse(JSON.stringify(doc))).sort();
	importer.openKitImport();
	const handle = last();
	handle.props.onConfirm(['kit:it_walk', 'kit:it_face'], [picked(byId('it_walk')), picked(byId('it_face'))]);
	close(handle);
	await settle();
	assert.deepEqual(assetsOf(), [
		['video', lib('up-video-2'), 'hands.mp4'],
		['image', lib('up-image-1'), 'face.png']
	]);
	assert.deepEqual(Object.keys(JSON.parse(JSON.stringify(doc))).sort(), before);
	assert.ok(doc.assets.every((a) => Object.keys(a.origin).join() === 'kind,itemId'));
	assert.deepEqual(toasts, [['success', '2 files imported']]);
});

test('the whole kit: its accepted media in kit order, then a warning that says how many were left and why', async () => {
	importer.openKitImport();
	const handle = last();
	handle.props.kits.onLoad([MARA]);
	close(handle);
	await settle();
	assert.deepEqual(assetsOf().map(([kind, origin]) => [kind, origin.itemId]), [
		['image', 'up-image-1'],
		['image', 'up-image-3'],
		['video', 'up-video-2']
	]);
	assert.deepEqual(toasts, [
		['success', '3 files imported'],
		['warning', '3 kit items left out: 1 text, 1 file deleted from the library, 1 of a kind this place does not take']
	]);
});

test('picked items and a whole kit confirmed together: picked first, then the kit, and a file reached twice is one asset', async () => {
	importer.openKitImport();
	const handle = last();
	handle.props.onConfirm(['kit:it_coat'], [picked(byId('it_coat'))]);
	handle.props.kits.onLoad([MARA]);
	close(handle);
	await settle();
	assert.deepEqual(doc.assets.map((a) => a.origin.itemId), ['up-image-3', 'up-image-1', 'up-video-2']);
	assert.equal(resolved.length, 3);
});

test('a kit the host could not load leaves the picker open, and the next confirm imports what it brings, not only the first one', async () => {
	importer.openKitImport();
	const handle = last();
	handle.props.onConfirm(['kit:it_coat'], [picked(byId('it_coat'))]);
	await settle();
	assert.deepEqual(doc.assets.map((a) => a.origin.itemId), ['up-image-3']);
	handle.props.kits.onLoad([MARA]);
	close(handle);
	await settle();
	assert.deepEqual(doc.assets.map((a) => a.origin.itemId), ['up-image-3', 'up-image-1', 'up-video-2']);
	assert.equal(ui.hostModal, false);
});

test('a file already in the project, from the Library or from a kit, stays one asset and is counted already present', async () => {
	replaceDoc({ ...emptyDoc((key) => key), assets: [newAsset({ id: 'have', kind: 'image', name: 'hands.png', src: '/api/media/uploads/hands.png', origin: lib('up-image-1') })] });
	importer.openKitImport();
	const handle = last();
	handle.props.onConfirm(['kit:it_face', 'kit:it_coat'], [picked(byId('it_face')), picked(byId('it_coat'))]);
	close(handle);
	await settle();
	assert.deepEqual(doc.assets.map((a) => [a.id === 'have' ? 'have' : 'new', a.origin.itemId]), [['have', 'up-image-1'], ['new', 'up-image-3']]);
	assert.deepEqual(toasts, [['success', '1 file imported · 1 file already in the project']]);
});

test('a kit item whose library file the server no longer resolves is counted like a library file that fails', async () => {
	importer.openKitImport();
	const handle = last();
	handle.props.onConfirm(['kit:it_x'], [picked(item({ id: 'it_x', media_ref: 'gone-1' }))]);
	close(handle);
	await settle();
	assert.equal(doc.assets.length, 0);
	assert.deepEqual(toasts, [['warning', '1 file could not be imported']]);
});

test('closing the kit picker without choosing imports nothing and says nothing', async () => {
	importer.openKitImport();
	close(last());
	await settle();
	assert.equal(doc.assets.length, 0);
	assert.deepEqual(toasts, []);
	assert.equal(ui.hostModal, false);
});

test('a single slot (reference media, first image, a video to cut): one item, single select, and the slot gets its asset id', async () => {
	const got = [];
	importer.pickKitMedia('video', 'Shot reference', (id) => got.push(id), () => got.push('dismissed'));
	const handle = last();
	assert.equal(handle.props.multiple, undefined);
	assert.equal(handle.props.closeOnPick, false);
	assert.deepEqual(handle.props.mediaTypes, ['video']);
	assert.equal(handle.props.kits, undefined);
	handle.props.onPick(picked(byId('it_walk')));
	assert.equal(handle.open, false);
	await settle();
	assert.equal(got.length, 1);
	assert.equal(doc.assets.find((a) => a.id === got[0]).origin.itemId, 'up-video-2');
});

test('a single slot refuses a text, a deleted file and a wrong kind with a warning, and the picker stays open for another choice', async () => {
	const got = [];
	importer.pickKitMedia('image', 'Card', (id) => got.push(id), () => got.push('dismissed'));
	const handle = last();
	handle.props.onPick(picked(byId('it_bio')));
	handle.props.onPick(picked(byId('it_gone')));
	handle.props.onPick(picked(byId('it_walk')));
	await settle();
	assert.equal(handle.open, true);
	assert.deepEqual(got, []);
	assert.deepEqual(toasts, [
		['warning', '“bio” is a text, not a media file.'],
		['warning', '“gone” has no file: it was deleted from the library.'],
		['warning', '“walk” is of a kind this place does not take.']
	]);
	handle.props.onPick(picked(byId('it_coat')));
	await settle();
	assert.equal(got.length, 1);
	close(handle);
	assert.equal(got.length, 1);
});

test('a single slot dismissed: the caller hears it, as with the Library picker', () => {
	const got = [];
	importer.pickKitMedia('video', 'Cut', () => got.push('picked'), () => got.push('dismissed'));
	close(last());
	assert.deepEqual(got, ['dismissed']);
});

test('a place that adds (a bible card, a new reference or sound of a shot): many items or the whole kit, and every asset id reaches it in order', async () => {
	const got = [];
	importer.pickKitMany('image', 'Add an image', (ids) => got.push(...ids), () => got.push('dismissed'));
	const handle = last();
	assert.equal(handle.props.multiple, true);
	assert.deepEqual(handle.props.mediaTypes, ['image']);
	handle.props.kits.onLoad([MARA]);
	close(handle);
	await settle();
	assert.deepEqual(got.map((id) => doc.assets.find((a) => a.id === id).origin.itemId), ['up-image-1', 'up-image-3']);
	assert.deepEqual(toasts, [['warning', '4 kit items left out: 1 text, 1 file deleted from the library, 2 of a kind this place does not take']]);
});

test('a place that adds, with a file the project already has: the existing asset id is given, no new asset', async () => {
	replaceDoc({ ...emptyDoc((key) => key), assets: [newAsset({ id: 'have', kind: 'audio', name: 'voice.wav', src: '/x.wav', origin: lib('up-audio-5') })] });
	const got = [];
	importer.pickKitMany('audio', 'New sound', (ids) => got.push(...ids));
	const handle = last();
	handle.props.onConfirm(['kit:it_voice'], [picked(byId('it_voice'))]);
	close(handle);
	await settle();
	assert.deepEqual(got, ['have']);
	assert.equal(doc.assets.length, 1);
	assert.equal(resolved.length, 0);
});

test('a place that adds, where the kit has nothing it takes: nothing is added and the caller hears it was dismissed', async () => {
	const got = [];
	importer.pickKitMany('audio', 'New sound', (ids) => got.push(...ids), () => got.push('dismissed'));
	const handle = last();
	handle.props.kits.onLoad([{ ...MARA, items: MARA.items.filter((it) => it.kind !== 'audio') }]);
	close(handle);
	await settle();
	assert.deepEqual(got, ['dismissed']);
	assert.equal(doc.assets.length, 0);
});

test('a host without MediaPickerModal: no kit entry, and the Library picker opens as before, with the same props', () => {
	delete components.MediaPickerModal;
	assert.equal(importer.kitsAvailable(), false);
	assert.equal(importer.libraryAvailable(), true);
	importer.openKitImport();
	assert.equal(mounted.length, 0);
	assert.deepEqual(toasts, [['warning', 'The media picker is not available on this server.']]);
	importer.openImport();
	const { name, props } = last();
	assert.equal(name, 'UploadLibraryModal');
	assert.deepEqual(Object.keys(props).sort(), ['isOpen', 'mediaType', 'multiple', 'onClose', 'onConfirm', 'selectableMediaTypes', 'selectedKeys', 'title']);
	assert.deepEqual(props.selectableMediaTypes, ['image', 'video']);
});

test('the Library picker of a single slot still opens UploadLibraryModal with onSelect, where the kit picker is offered', () => {
	importer.pickMedia('image', 'Card', () => {});
	assert.equal(last().name, 'UploadLibraryModal');
	assert.equal(typeof last().props.onSelect, 'function');
	assert.equal(last().props.sources, undefined);
});

test('on the seeded kits, the whole-kit import builds the assets of tests/fixtures/kit_picks.json, which the server schema accepts', async () => {
	const cases = seededKits();
	for (const [n, { kit, allowed }] of cases.entries()) {
		replaceDoc(emptyDoc((key) => key));
		toasts.length = 0;
		const kind = allowed.length > 1 ? 'media' : allowed[0];
		if (kind === 'media' || kind === 'audio') importer.openKitImport(kind);
		else importer.pickKitMany(kind, 'Place', () => {});
		const handle = last();
		handle.props.kits.onLoad([kit]);
		close(handle);
		await settle();
		const strip = (assets) => JSON.parse(JSON.stringify(assets)).map(({ id, ...rest }) => rest);
		assert.deepEqual(strip(doc.assets), strip(fixture.cases[n].assets), `kit ${n}`);
	}
});
