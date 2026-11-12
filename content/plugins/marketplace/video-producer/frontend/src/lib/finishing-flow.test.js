import { test, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';

const hooks = `
import { readFile } from 'node:fs/promises';
import { compileModule } from ${JSON.stringify(import.meta.resolve('svelte/compiler'))};
export async function load(url, context, next) {
	if (url.endsWith('.svelte.js')) return { format: 'module', shortCircuit: true, source: compileModule(await readFile(new URL(url), 'utf8'), { generate: 'client', filename: url }).js.code };
	if (url.endsWith('.json')) return { format: 'module', shortCircuit: true, source: 'export default ' + (await readFile(new URL(url), 'utf8')) };
	return next(url, context);
}`;
register(`data:text/javascript,${encodeURIComponent(hooks)}`, import.meta.url);

globalThis.window = { __potionui: { confirm: async () => true, notifications: { toast() {} } } };

const { doc, ui, undo, dispose } = await import('./state.svelte.js');
const { bindClient, openProject, project } = await import('./project.svelte.js');
const { exportState, startVideoExport, pollJob, resumeExports, disposeExports } = await import('./exports.svelte.js');
const { finishState, finishTiming, finishJob, cancelFinish, loadFinishing, setFinish, downloadFinished, resetFinishing, finishShown } = await import('./finishing.svelte.js');
const { finishSteps, finishingOf, toolsById } = await import('./finishing.js');
const { normalizeTool } = await import('./api.js');
const { emptyDoc, newClip } = await import('./starter.js');
const { setLocale } = await import('../i18n/index.svelte.js');

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
const copy = (value) => JSON.parse(JSON.stringify(value));

const TOOLS = [
	{ id: 'film_grain', params: [{ name: 'amount', default: 30, min: 0, max: 100, step: 1 }, { name: 'size', default: 1.5, min: 1, max: 3, step: 0.1 }] },
	{ id: 'vignette', params: [{ name: 'amount', default: 40, min: 0, max: 100, step: 1 }, { name: 'radius', default: 50, min: 0, max: 100, step: 1 }] },
	{ id: 'fix_colors', params: [{ name: 'strength', default: 100, min: 0, max: 100, step: 5 }] },
	{ id: 'lut', params: [{ name: 'lut', type: 'lut', options: [{ label: 'None', value: '' }, { label: 'warm', value: 'warm.cube' }] }, { name: 'strength', default: 1, min: 0, max: 1, step: 0.05 }] }
].map(normalizeTool);

function makeServer(script = {}) {
	const calls = [];
	const stored = { doc: null, version: 1 };
	const exportJob = (status, extra = {}) => ({ id: 'job1', projectId: 'p1', format: script.format ?? 'web', resolution: '720p', ratio: null, stems: false, status, progress: status === 'done' ? 100 : 10, step: null, file: status === 'done' ? { name: 'Film_720p.mp4', size: 10, url: null } : null, media: status === 'done' && (script.format ?? 'web') !== 'master' ? { kind: 'library', itemId: 'item-raw', url: '/api/media/uploads/raw.mp4' } : null, error: null, createdAt: '2026-11-12T10:00:00Z', finishedAt: null, ...extra });
	const client = {
		async getProject(id) {
			return { ok: true, data: { id, name: 'Film', version: stored.version, doc: copy(stored.doc), createdAt: null, updatedAt: null, unchanged: false } };
		},
		async saveProject(id, next) {
			stored.doc = copy(next);
			stored.version += 1;
			calls.push(['save']);
			return { ok: true, data: { id, name: 'Film', version: stored.version, doc: null, createdAt: null, updatedAt: null, unchanged: false } };
		},
		saveLastProject() {},
		async listTakes() {
			return { ok: true, data: [] };
		},
		async listRenditions() {
			return { ok: true, data: [] };
		},
		async listExports() {
			return { ok: true, data: [] };
		},
		async exportVideo(projectId, body) {
			calls.push(['export', body]);
			return { ok: true, data: exportJob('queued') };
		},
		async exportJob() {
			calls.push(['poll']);
			return { ok: true, data: exportJob(script.exportEnds ?? 'done') };
		},
		async mediaTools(kind) {
			calls.push(['tools', kind]);
			return script.tools === false ? { ok: false, error: { status: 500, code: 'server_error' } } : { ok: true, data: TOOLS };
		},
		async startMediaRun(body) {
			calls.push(['run', body]);
			return script.start ?? { ok: true, data: { id: 'run1' } };
		},
		async mediaRun(id) {
			calls.push(['read', id]);
			const next = (script.reads ?? []).shift() ?? { status: 'completed', progress: 1, items: [{ id: 'item-fin', name: 'Film_720p-grain.mp4', mediaType: 'video', url: '/api/media/uploads/fin.mp4' }] };
			return { ok: true, data: { id, label: '', items: [], error: null, progress: 0, ...next } };
		},
		async cancelMediaRun(id) {
			calls.push(['cancel', id]);
			return { ok: true, data: { id, status: 'cancelled', progress: 0, label: '', items: [], error: null } };
		},
		async downloadCore(path, filename) {
			calls.push(['download', path, filename]);
			return { ok: true, data: { filename } };
		}
	};
	return { calls, client, stored };
}

let server = null;

async function open(script) {
	server = makeServer(script);
	const start = emptyDoc((key) => key);
	start.clips = [newClip({ id: 's1', label: 'One', start: 0, dur: 4 })];
	if (script?.finishing) start.master.finishing = script.finishing;
	server.stored.doc = start;
	bindClient(server.client);
	resetFinishing();
	await openProject('p1');
	await resumeExports();
	await loadFinishing();
}

async function exportAndFinish(format = 'web') {
	const steps = finishSteps(finishingOf(doc), toolsById(finishState.tools));
	await startVideoExport({ format, resolution: '720p', finish: steps });
	await pollJob();
	for (let i = 0; i < 50 && finishState.run?.status === 'running'; i++) await settle();
	return steps;
}

beforeEach(() => {
	setLocale('en');
	finishTiming.wait = async () => {};
	ui.modal = null;
});

after(() => {
	disposeExports();
	dispose();
	delete globalThis.window;
});

test('finishing flow: a web export with grain and vignette is finished by one run on its library item', async () => {
	await open({ finishing: { grain: { on: true, amount: 50, size: 2 }, vignette: { on: true } } });
	const steps = await exportAndFinish();
	assert.deepEqual(steps.map((x) => x.tool), ['vignette', 'film_grain']);
	const run = server.calls.find(([kind]) => kind === 'run');
	assert.deepEqual(run[1], { steps: [{ tool: 'vignette', params: { amount: 40, radius: 50 } }, { tool: 'film_grain', params: { amount: 50, size: 2 } }], inputs: [{ item_id: 'item-raw' }], output: { mode: 'new' } });
	assert.deepEqual(server.calls.find(([kind]) => kind === 'export')[1], { format: 'web', resolution: '720p', stems: false });
	assert.equal(finishState.run.status, 'done');
	assert.equal(finishState.run.item.id, 'item-fin');
	assert.equal(finishState.run.fileName, 'Film_720p.mp4');
	assert.equal(exportState.job.status, 'done');
	assert.equal(exportState.job.media.itemId, 'item-raw');
	await downloadFinished();
	assert.deepEqual(server.calls.at(-1), ['download', '/api/media/uploads/fin.mp4', 'Film_720p_finished.mp4']);
	assert.equal(ui.toasts.at(-1).message, 'Finished film ready: Film_720p_finished.mp4');
});

test('finishing flow: nothing switched on means a plain export and no run', async () => {
	await open({});
	const steps = await exportAndFinish();
	assert.deepEqual(steps, []);
	assert.equal(server.calls.some(([kind]) => kind === 'run'), false);
	assert.equal(finishShown(), null);
});

test('finishing flow: a master export is never finished, whatever is switched on', async () => {
	await open({ format: 'master', finishing: { grain: { on: true } } });
	await startVideoExport({ format: 'master', resolution: '720p', finish: finishSteps(finishingOf(doc), toolsById(finishState.tools)) });
	await pollJob();
	await settle();
	assert.equal(server.calls.some(([kind]) => kind === 'run'), false);
	assert.equal(finishShown(), null);
});

test('finishing flow: a failed or cancelled export starts no finishing', async () => {
	await open({ exportEnds: 'failed', finishing: { grain: { on: true } } });
	await exportAndFinish();
	assert.equal(server.calls.some(([kind]) => kind === 'run'), false);
	assert.equal(finishShown(), null);
});

test('finishing flow: a finishing that fails leaves the plain export done and says why', async () => {
	await open({ finishing: { lut: { on: true, ref: 'warm.cube', strength: 0.7 } }, reads: [{ status: 'failed', error: { code: 'tool_failed', message: 'ffmpeg failed' } }] });
	await exportAndFinish();
	assert.deepEqual(server.calls.find(([kind]) => kind === 'run')[1].steps, [{ tool: 'lut', params: { strength: 0.7, lut: 'warm.cube' } }]);
	assert.deepEqual([finishState.run.status, finishState.run.failure, finishState.run.message], ['failed', 'failed', 'ffmpeg failed']);
	assert.equal(exportState.job.status, 'done');
});

test('finishing flow: a full queue at the start is told as such', async () => {
	await open({ finishing: { grain: { on: true } }, start: { ok: false, error: { status: 429, code: 'too_many_queued', message: 'You already have 8 runs waiting' } } });
	await exportAndFinish();
	assert.deepEqual([finishState.run.status, finishState.run.failure], ['failed', 'busy']);
});

test('finishing flow: cancelling the finishing cancels the core run and keeps the plain export', async () => {
	await open({ finishing: { grain: { on: true } }, reads: [{ status: 'running', progress: 0.3 }, { status: 'running', progress: 0.6 }] });
	let reads = 0;
	finishTiming.wait = async () => {
		reads += 1;
		if (reads === 2) await cancelFinish();
	};
	await exportAndFinish();
	assert.equal(finishState.run.status, 'cancelled');
	assert.ok(server.calls.some(([kind, id]) => kind === 'cancel' && id === 'run1'));
	assert.equal(exportState.job.status, 'done');
});

test('finishing flow: the choices are kept in the project, one undo step each, and a project without them stays as it was', async () => {
	await open({});
	assert.equal('finishing' in doc.master, false);
	setFinish('grain', { on: true });
	assert.deepEqual(doc.master.finishing, { grain: { on: true, amount: 30, size: 1.5 } });
	setFinish('grain', { amount: 70 });
	setFinish('grain', { on: false });
	assert.deepEqual(doc.master.finishing, { grain: { on: false, amount: 70, size: 1.5 } });
	undo();
	assert.deepEqual(doc.master.finishing.grain.on, true);
	undo();
	undo();
	assert.equal('finishing' in doc.master, false);
});

test('finishing flow: without the core catalog nothing is offered and an export stays plain', async () => {
	await open({ tools: false, finishing: { grain: { on: true } } });
	assert.deepEqual(finishState.tools, []);
	const steps = await exportAndFinish();
	assert.deepEqual(steps, []);
	assert.equal(server.calls.some(([kind]) => kind === 'run'), false);
});

test('finishing flow: a job finished without a plan is left alone', async () => {
	await open({});
	assert.equal(await finishJob({ id: 'other', status: 'done', format: 'web', media: { itemId: 'x' } }), null);
	assert.equal(project.id, 'p1');
});
