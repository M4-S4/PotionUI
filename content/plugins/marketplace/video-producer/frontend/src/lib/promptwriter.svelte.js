import { getClient } from './project.svelte.js';
import { WRITE_TIMEOUT_MS, canUndo, canWrite, writeBody, writerIdle, writerStep } from './promptwriter.js';

export const STATUS_TTL_MS = 60000;

const known = { status: null, at: 0 };

export function forgetWriterStatus() {
	known.status = null;
	known.at = 0;
}

export function createWriter({ client = getClient, timeout = WRITE_TIMEOUT_MS, now = Date.now } = {}) {
	const box = $state({ state: writerIdle(), status: known.status, checking: false });
	let controller = null;

	const dispatch = (event) => {
		box.state = writerStep(box.state, event);
	};

	async function check({ force = false } = {}) {
		if (!force && known.status && now() - known.at < STATUS_TTL_MS) {
			box.status = known.status;
			return box.status;
		}
		const api = client();
		if (typeof api?.promptWriter !== 'function') return box.status;
		box.checking = true;
		const result = await api.promptWriter();
		box.checking = false;
		const status = result.ok ? result.data : { available: false, vision: false, name: null, code: result.error?.code || 'writer_unavailable', message: result.error?.message || null };
		if (result.ok) {
			known.status = status;
			known.at = now();
		}
		box.status = status;
		return status;
	}

	async function write({ projectId, shotId, option, doc }) {
		const api = client();
		if (!option || !projectId || typeof api?.writePrompt !== 'function' || !canWrite(box.state, box.status)) return;
		controller?.abort();
		dispatch({ type: 'start' });
		const run = box.state.run;
		const mine = new AbortController();
		controller = mine;
		const timer = setTimeout(() => {
			mine.abort();
			dispatch({ type: 'fail', run, error: { code: 'write_timeout', message: '' } });
		}, timeout);
		const result = await api.writePrompt(projectId, shotId, writeBody(box.state.request, option, doc), mine.signal);
		clearTimeout(timer);
		if (controller === mine) controller = null;
		if (result.ok) dispatch({ type: 'done', run, written: result.data });
		else if (result.error?.code !== 'aborted') dispatch({ type: 'fail', run, error: result.error });
	}

	function cancel() {
		controller?.abort();
		controller = null;
		dispatch({ type: 'cancel' });
	}

	function close() {
		controller?.abort();
		controller = null;
		dispatch({ type: 'close' });
	}

	function accept(current) {
		const text = box.state.proposal?.text ?? null;
		if (text === null) return null;
		dispatch({ type: 'accept', current });
		return text;
	}

	function undo(current) {
		if (!canUndo(box.state, current)) return null;
		const before = box.state.before;
		dispatch({ type: 'undo' });
		return before;
	}

	return {
		get state() {
			return box.state;
		},
		get status() {
			return box.status;
		},
		get checking() {
			return box.checking;
		},
		get busy() {
			return box.state.phase === 'writing';
		},
		ready: () => canWrite(box.state, box.status),
		undoable: (current) => canUndo(box.state, current),
		open: () => dispatch({ type: 'open' }),
		edit: (request) => dispatch({ type: 'edit', request }),
		reject: () => dispatch({ type: 'reject' }),
		forget: () => dispatch({ type: 'forget' }),
		check,
		write,
		cancel,
		close,
		accept,
		undo
	};
}
