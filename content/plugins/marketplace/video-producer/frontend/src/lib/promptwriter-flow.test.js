import { test, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { hostUis, renderComponent, textOf } from './svelte-harness.js';

const { doc, ui, dispose } = await import('./state.svelte.js');
const { project, bindClient } = await import('./project.svelte.js');
const { createWriter, forgetWriterStatus } = await import('./promptwriter.svelte.js');
const { t } = await import('../i18n/index.svelte.js');

const BUTTON = new URL('../components/shots/WritePromptButton.svelte', import.meta.url).href;
const PANEL = new URL('../components/shots/WritePromptPanel.svelte', import.meta.url).href;
const SHOT_EDITOR = new URL('../components/modals/ShotEditor.svelte', import.meta.url).href;
const READY = { available: true, vision: true, name: 'Qwen3-VL', code: null, message: null };
const OPTION = { preset: 'MiniMax-H3-12GB', model: null };
const settle = async () => {
	for (let i = 0; i < 5; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
};

after(() => dispose());
beforeEach(() => forgetWriterStatus());

function fakeClient({ status = READY, answer = null } = {}) {
	const calls = { status: 0, writes: [] };
	const client = {
		async promptWriter() {
			calls.status += 1;
			return typeof status === 'function' ? status() : { ok: true, data: status };
		},
		async writePrompt(projectId, shotId, body, signal) {
			calls.writes.push({ projectId, shotId, body, signal });
			if (answer) return answer(signal);
			return { ok: true, data: { text: 'The fisherman lights his lantern.', name: 'Qwen3-VL', images: 2, imagesSkipped: 0 } };
		}
	};
	return { client, calls };
}

const aborting = (signal) => new Promise((resolve) => signal.addEventListener('abort', () => resolve({ ok: false, error: { status: 0, code: 'aborted', message: '', data: null } })));

async function asked(client, request = 'the lantern', options = {}) {
	const writer = createWriter({ client: () => client, ...options });
	await writer.check();
	writer.open();
	writer.edit(request);
	return writer;
}

test('the status is asked once a minute, and a failed check reads as no writer', async () => {
	const { client, calls } = fakeClient();
	const writer = createWriter({ client: () => client });
	assert.equal(writer.status, null);
	await writer.check();
	assert.deepEqual(writer.status, READY);
	await createWriter({ client: () => client }).check();
	assert.equal(calls.status, 1);
	await writer.check({ force: true });
	assert.equal(calls.status, 2);
	forgetWriterStatus();
	const down = fakeClient({ status: () => ({ ok: false, error: { status: 503, code: 'unavailable', message: 'Down.' } }) });
	const blind = createWriter({ client: () => down.client });
	await blind.check();
	assert.deepEqual([blind.status.available, blind.status.code, blind.status.message], [false, 'unavailable', 'Down.']);
	assert.equal(blind.ready(), false);
});

test('writing sends the request for the shot, shows the proposal, and accept then undo give back the prompt of before', async () => {
	const { client, calls } = fakeClient();
	const writer = await asked(client);
	const source = { clips: [{ id: 's1' }] };
	await writer.write({ projectId: 'p1', shotId: 's1', option: OPTION, doc: source });
	assert.equal(calls.writes.length, 1);
	assert.deepEqual([calls.writes[0].projectId, calls.writes[0].shotId], ['p1', 's1']);
	assert.deepEqual(calls.writes[0].body, { request: 'the lantern', preset_id: 'MiniMax-H3-12GB', model_id: null, doc: source });
	assert.equal(writer.state.phase, 'proposed');
	const placed = writer.accept('My own words.');
	assert.equal(placed, 'The fisherman lights his lantern.');
	assert.equal(writer.undoable(placed), true);
	assert.equal(writer.undo('Edited after.'), null);
	assert.equal(writer.undo(placed), 'My own words.');
	assert.equal(writer.undo(placed), null);
});

test('without a request, a model or a project nothing is sent', async () => {
	const { client, calls } = fakeClient();
	const writer = await asked(client, '  ');
	await writer.write({ projectId: 'p1', shotId: 's1', option: OPTION, doc: {} });
	writer.edit('the lantern');
	await writer.write({ projectId: null, shotId: 's1', option: OPTION, doc: {} });
	await writer.write({ projectId: 'p1', shotId: 's1', option: null, doc: {} });
	forgetWriterStatus();
	const none = fakeClient({ status: { available: false, vision: false, name: null, code: 'no_llm', message: 'None.' } });
	const off = await asked(none.client);
	await off.write({ projectId: 'p1', shotId: 's1', option: OPTION, doc: {} });
	assert.equal(calls.writes.length + none.calls.writes.length, 0);
});

test('cancel stops the wait and aborts the request, and its late answer is dropped', async () => {
	const { client, calls } = fakeClient({ answer: aborting });
	const writer = await asked(client);
	const pending = writer.write({ projectId: 'p1', shotId: 's1', option: OPTION, doc: {} });
	assert.equal(writer.state.phase, 'writing');
	assert.equal(writer.busy, true);
	writer.cancel();
	await pending;
	assert.equal(calls.writes[0].signal.aborted, true);
	assert.deepEqual([writer.state.phase, writer.state.error, writer.state.proposal], ['open', null, null]);
});

test('a writing past the time limit fails with its own message and the request can be sent again', async () => {
	const { client, calls } = fakeClient({ answer: aborting });
	const writer = await asked(client, 'the lantern', { timeout: 5 });
	await writer.write({ projectId: 'p1', shotId: 's1', option: OPTION, doc: {} });
	assert.equal(calls.writes[0].signal.aborted, true);
	assert.deepEqual([writer.state.phase, writer.state.error.code], ['failed', 'write_timeout']);
	assert.equal(writer.ready(), true);
});

test('a refusal of the core is shown with its code', async () => {
	const { client } = fakeClient({ answer: async () => ({ ok: false, error: { status: 422, code: 'vision_not_supported', message: 'Cannot see.', data: null } }) });
	const writer = await asked(client);
	await writer.write({ projectId: 'p1', shotId: 's1', option: OPTION, doc: {} });
	assert.deepEqual([writer.state.phase, writer.state.error.code], ['failed', 'vision_not_supported']);
	const html = await renderComponent(PANEL, { writer, text: '', frames: 2 });
	assert.match(html, /data-writer-error="vision_not_supported"/);
	assert.ok(textOf(html).includes(t('modals.shot.writer.error.vision_not_supported')));
	assert.ok(hostUis(html, 'Button').some((props) => props.disabled === false));
});

test('the button is grey with its reason before any click, and opens the panel once a model is there', async () => {
	const none = fakeClient({ status: { available: false, vision: false, name: null, code: 'no_llm', message: 'None.' } });
	const off = createWriter({ client: () => none.client });
	await off.check();
	const grey = await renderComponent(BUTTON, { writer: off });
	assert.match(grey, /data-write-prompt="off"/);
	assert.match(grey, /aria-disabled="true"/);
	assert.equal(hostUis(grey, 'Tooltip')[0].text, t('modals.shot.writer.error.no_llm'));
	off.open();
	assert.equal(off.state.phase, 'open');
	forgetWriterStatus();
	const { client } = fakeClient();
	const on = createWriter({ client: () => client });
	await on.check();
	const ready = await renderComponent(BUTTON, { writer: on });
	assert.match(ready, /data-write-prompt="on"/);
	assert.equal(hostUis(ready, 'Tooltip')[0].text, t('modals.shot.writer.tip', { name: 'Qwen3-VL' }));
	assert.match(await renderComponent(BUTTON, { writer: on, saved: false }), /data-write-prompt="off"/);
	assert.equal(hostUis(await renderComponent(BUTTON, { writer: on, engine: false }), 'Tooltip')[0].text, t('modals.shot.composedNone'));
});

test('the panel shows the wait with a cancel, then the proposal with accept and discard, then the undo line', async () => {
	let release;
	const { client } = fakeClient({ answer: () => new Promise((resolve) => (release = resolve)) });
	const writer = await asked(client);
	assert.doesNotMatch(await renderComponent(PANEL, { writer: createWriter({ client: () => client }), text: '' }), /data-writer/);
	const opened = await renderComponent(PANEL, { writer, text: 'Old.', frames: 2 });
	assert.match(opened, /data-writer="open"/);
	assert.ok(textOf(opened).includes(t('modals.shot.writer.sourcesFrames', { name: 'Qwen3-VL', count: 2 })));
	const pending = writer.write({ projectId: 'p1', shotId: 's1', option: OPTION, doc: {} });
	const waiting = await renderComponent(PANEL, { writer, text: 'Old.', frames: 2 });
	assert.match(waiting, /data-writer-waiting/);
	assert.ok(hostUis(waiting, 'Button').length === 1);
	assert.ok(textOf(waiting).includes(t('modals.shot.writer.cancel')));
	release({ ok: true, data: { text: 'The fisherman lights his lantern.', name: 'Qwen3-VL', images: 2, imagesSkipped: 0 } });
	await pending;
	const proposed = await renderComponent(PANEL, { writer, text: 'Old.', frames: 2 });
	assert.match(proposed, /data-writer-proposal[^>]*>The fisherman lights his lantern\.</);
	assert.ok(textOf(proposed).includes(t('modals.shot.writer.accept')));
	assert.ok(textOf(proposed).includes(t('modals.shot.writer.reject')));
	assert.ok(textOf(proposed).includes(t('modals.shot.writer.replaces')));
	const placed = writer.accept('Old.');
	const applied = await renderComponent(PANEL, { writer, text: placed, frames: 2 });
	assert.match(applied, /data-writer-applied/);
	assert.ok(textOf(applied).includes(t('modals.shot.writer.undo')));
	assert.doesNotMatch(await renderComponent(PANEL, { writer, text: `${placed} More.`, frames: 2 }), /data-writer-applied/);
});

test('a model that cannot see says the shot images are not sent', async () => {
	const { client } = fakeClient({ status: { ...READY, vision: false } });
	const writer = await asked(client);
	const html = await renderComponent(PANEL, { writer, text: '', frames: 2 });
	assert.match(html, /data-writer-blind/);
	assert.ok(textOf(html).includes(t('modals.shot.writer.blind', { name: 'Qwen3-VL', count: 2 })));
	assert.ok(textOf(html).includes(t('modals.shot.writer.sources', { name: 'Qwen3-VL' })));
});

test('the shot editor carries the button beside the prompt library and the card words of a shot that sets its own', async () => {
	const { client } = fakeClient();
	bindClient(client);
	project.id = 'p1';
	ui.me = null;
	doc.assets = [];
	doc.clips = [{ id: 's1', start: 0, dur: 4, label: 'One', stage: 'todo' }];
	doc.prompts = [{ id: 'p1', shotId: 's1', start: 0, dur: 4, strength: 75, text: 'A quay.' }];
	doc.frames = [];
	const plain = await renderComponent(SHOT_EDITOR, { id: 's1' });
	assert.match(plain, /data-write-prompt=/);
	assert.ok(textOf(plain).includes(t('modals.shot.writer.button')));
	assert.doesNotMatch(plain, /data-shot-card-words/);
	doc.clips[0].cardWords = 'label';
	const own = await renderComponent(SHOT_EDITOR, { id: 's1' });
	assert.match(own, /data-shot-card-words="label"/);
	assert.ok(textOf(own).includes(t('modals.shot.asProject', { value: t('project.cardWords.card') })));
	assert.ok(textOf(own).includes(t('modals.shot.cardWords.unused')));
	await settle();
	delete doc.clips[0].cardWords;
	project.id = null;
});

test('a proposal that dropped a reference video tag says which one before it is used', async () => {
	const { client } = fakeClient({ answer: async () => ({ ok: true, data: { text: 'The fisherman walks along the quay.', name: 'Qwen3-VL', images: 0, imagesSkipped: 0, lostTags: ['@vref1', '@vref2'] } }) });
	const writer = await asked(client);
	await writer.write({ projectId: 'p1', shotId: 's1', option: OPTION, doc: {} });
	const html = await renderComponent(PANEL, { writer, text: 'She walks past @vref1 and @vref2.', frames: 0 });
	assert.match(html, /data-writer-lost-tags="@vref1 @vref2"/);
	assert.ok(textOf(html).includes(t('modals.shot.writer.lostTags', { tags: '@vref1, @vref2', count: 2 })));
	const kept = createWriter({ client: () => fakeClient().client });
	await kept.check();
	kept.open();
	kept.edit('x');
	await kept.write({ projectId: 'p1', shotId: 's1', option: OPTION, doc: {} });
	assert.doesNotMatch(await renderComponent(PANEL, { writer: kept, text: '', frames: 0 }), /data-writer-lost-tags/);
});
