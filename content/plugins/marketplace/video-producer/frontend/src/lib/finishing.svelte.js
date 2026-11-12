import { t } from '../i18n/index.svelte.js';
import { doc, ui, act, toast } from './state.svelte.js';
import { getClient, project } from './project.svelte.js';
import { FINISH_POLL_MS, cleanFinishing, effectByKey, finishFailure, finishRequest, finishable, finishedName, finishingOf, settingOf, toolsById } from './finishing.js';

export const finishState = $state({ tools: null, loading: false, run: null });

const planned = new Map();
let ticket = 0;

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export const finishTiming = { wait: delay };

export const finishTools = () => toolsById(finishState.tools ?? []);

export const finishShown = () => (finishState.run?.projectId === project.id ? finishState.run : null);

export async function loadFinishing() {
	const client = getClient();
	if (!client || finishState.loading || finishState.tools) return;
	finishState.loading = true;
	const result = await client.mediaTools('video');
	finishState.loading = false;
	finishState.tools = result.ok ? result.data : [];
}

export function setFinish(key, patch) {
	const effect = effectByKey(key);
	if (!effect) return;
	act(() => {
		const all = { ...finishingOf(doc) };
		all[key] = { ...settingOf(all, effect, finishTools().get(effect.tool)), ...patch };
		const clean = cleanFinishing(all);
		if (clean) doc.master.finishing = clean;
		else delete doc.master.finishing;
	});
}

export function planFinish(jobId, steps, fileName) {
	if (!jobId || !steps?.length) return;
	planned.set(jobId, { steps, fileName, projectId: project.id });
}

export const finishPlanned = (jobId) => planned.has(jobId);

export function dropFinish(jobId) {
	planned.delete(jobId);
}

function ended(status, extra = {}) {
	finishState.run = { ...finishState.run, status, runId: null, ...extra };
}

export async function finishJob(job, { wait = finishTiming.wait } = {}) {
	const plan = planned.get(job?.id);
	if (!plan || plan.projectId !== project.id) return null;
	planned.delete(job.id);
	const client = getClient();
	const mine = ++ticket;
	const fileName = !job.stems && job.file?.name ? job.file.name : plan.fileName;
	finishState.run = { jobId: job.id, projectId: plan.projectId, runId: null, status: 'running', progress: 0, item: null, failure: null, message: '', fileName };
	if (!finishable(job) || !client) return ended('failed', { failure: 'noLibrary' });
	const started = await client.startMediaRun(finishRequest(job.media.itemId, plan.steps));
	if (mine !== ticket) {
		if (started.ok) await client.cancelMediaRun(started.data.id);
		return null;
	}
	if (!started.ok) return ended('failed', { failure: finishFailure(started.error), message: started.error?.message ?? '' });
	finishState.run.runId = started.data.id;
	while (true) {
		await wait(FINISH_POLL_MS);
		if (mine !== ticket) return null;
		const read = await client.mediaRun(started.data.id);
		if (mine !== ticket) return null;
		if (!read.ok) return ended('failed', { failure: finishFailure(read.error), message: read.error?.message ?? '' });
		const run = read.data;
		finishState.run.progress = run.progress;
		if (run.status === 'failed') return ended('failed', { failure: finishFailure(run.error), message: run.error?.message ?? '' });
		if (run.status === 'cancelled') return ended('cancelled');
		if (run.status === 'completed') {
			const item = run.items.find((x) => x.mediaType === 'video') ?? null;
			if (!item) return ended('failed', { failure: 'failed' });
			ended('done', { progress: 1, item });
			if (ui.modal?.type !== 'export') toast(t('exports.finish.toastDone', { file: finishedName(fileName) }), 'success', { label: t('exports.finish.download'), run: downloadFinished }, 6000);
			return item;
		}
	}
}

export async function cancelFinish() {
	const runId = finishState.run?.runId;
	ticket += 1;
	if (finishState.run?.status === 'running') ended('cancelled');
	if (runId) await getClient()?.cancelMediaRun(runId);
}

export async function downloadFinished() {
	const run = finishState.run;
	if (run?.status !== 'done' || !run.item) return;
	const result = await getClient().downloadCore(run.item.url, finishedName(run.fileName));
	if (!result.ok) return toast(t('exports.finish.downloadFailed'), 'error');
	toast(t('modals.export.downloaded', { file: result.data.filename }), 'success');
}

export function finishFailureText(run) {
	if (run?.failure === 'busy') return t('exports.finish.failure.busy');
	if (run?.failure === 'quota') return t('exports.finish.failure.quota');
	if (run?.failure === 'gone') return t('exports.finish.failure.gone');
	if (run?.failure === 'noLibrary') return t('exports.finish.failure.noLibrary');
	return run?.message ? t('exports.finish.failure.failedWith', { message: run.message }) : t('exports.finish.failure.failed');
}

export function resetFinishing() {
	ticket += 1;
	planned.clear();
	Object.assign(finishState, { tools: null, loading: false, run: null });
}
