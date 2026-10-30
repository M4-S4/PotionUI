import type { RunHandle } from './runs';

export type RunStartListener = (handle: RunHandle) => void;

const startListeners = new Set<RunStartListener>();

export function onRunStarted(listener: RunStartListener): () => void {
	startListeners.add(listener);
	return () => startListeners.delete(listener);
}

export function emitRunStarted(handle: RunHandle): void {
	for (const listener of [...startListeners]) listener(handle);
}
