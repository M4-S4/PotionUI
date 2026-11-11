import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import { hostUis, renderComponent } from './svelte-harness.js';
import { CHAIN_BY, chainByOf, setChainBy, setShotChainBy } from './chain.js';

globalThis.Image = class {
	set src(value) {}
};

const { doc, ui, dispose } = await import('./state.svelte.js');
const { project } = await import('./project.svelte.js');
const { setTakes } = await import('./takes.svelte.js');
const { chooseShotChainBy } = await import('./chain.svelte.js');
const { noteText } = await import('./production.svelte.js');
const { t } = await import('../i18n/index.svelte.js');

const MODALS = new URL('../components/modals/Modals.svelte', import.meta.url).href;
const CHAIN_SWITCH = new URL('../components/shots/ChainSwitch.svelte', import.meta.url).href;

after(() => dispose());

test('a project chains by the last frame unless it or a shot chose the continuation', () => {
	const film = { clips: [{ id: 's1' }, { id: 's2' }] };
	assert.deepEqual(CHAIN_BY, ['frame', 'video']);
	assert.equal(chainByOf(film, film.clips[1]), 'frame');
	setChainBy(film, 'video');
	assert.equal(film.chainBy, 'video');
	assert.equal(chainByOf(film, film.clips[1]), 'video');
	setShotChainBy(film, film.clips[1], 'frame');
	assert.equal(film.clips[1].chainBy, 'frame');
	setShotChainBy(film, film.clips[1], 'video');
	assert.equal('chainBy' in film.clips[1], false);
	setChainBy(film, 'audio');
	assert.equal(film.chainBy, 'video');
	setChainBy(film, 'frame');
	assert.equal('chainBy' in film, false);
});

test('the project properties offer the chain by continuation while the chain is on', async () => {
	delete doc.chainBy;
	delete doc.firstFromPrevious;
	ui.modal = { type: 'project' };
	const control = async () => hostUis(await renderComponent(MODALS), 'SegmentedControl').find((item) => item.ariaLabel === t('shell.properties.chainBy'));
	const shown = await control();
	assert.deepEqual(
		shown.items.map((item) => [item.id, item.label]),
		[
			['frame', t('shell.properties.chainByFrame')],
			['video', t('shell.properties.chainByVideo')]
		]
	);
	assert.equal(shown.selected, 'frame');
	doc.chainBy = 'video';
	assert.equal((await control()).selected, 'video');
	doc.firstFromPrevious = false;
	assert.equal(await control(), undefined);
	delete doc.firstFromPrevious;
	delete doc.chainBy;
	ui.modal = null;
});

test('a chained FL2VA shot chooses how it chains, and the choice follows the project until it sets its own', async () => {
	project.id = 'p1';
	doc.direction = { ...(doc.direction ?? {}), mode: 'FL2VA' };
	doc.clips = [
		{ id: 's1', assetId: null, take: null, start: 0, dur: 4, srcIn: 0, stage: 'todo', label: 'One' },
		{ id: 's2', assetId: null, take: null, start: 4, dur: 4, srcIn: 0, stage: 'todo', label: 'Two' }
	];
	setTakes('p1', []);
	const control = async () => hostUis(await renderComponent(CHAIN_SWITCH, { clip: doc.clips[1] }), 'SegmentedControl').find((item) => item.ariaLabel === t('shell.properties.chainBy'));
	assert.equal((await control()).selected, 'frame');
	chooseShotChainBy(doc.clips[1], 'video');
	assert.equal(doc.clips[1].chainBy, 'video');
	assert.equal((await control()).selected, 'video');
	doc.direction.mode = 'Ref2VA';
	assert.equal(await control(), undefined);
	doc.direction.mode = 'FL2VA';
	doc.clips[1].firstFromPrevious = false;
	assert.equal(await control(), undefined);
});

test('the estimate says why a shot asked to continue starts from the last image', () => {
	for (const code of ['chain_continue_unsupported', 'chain_continue_still', 'chain_continue_short', 'chain_continue_last_frame', 'chain_continue_too_long', 'chain_continue_sound']) {
		const text = noteText({ code, message: 'server text', data: { shot_id: 's2' } });
		assert.notEqual(text, 'server text', code);
		assert.equal(text.split('{').length, 1, code);
	}
});
