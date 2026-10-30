import { onRunStarted } from '$lib/media-tools/runEvents';

export interface LaunchTracker {
	delivered(): void;
	closed(): void;
}

export function trackLaunch(onAbandon: () => void): LaunchTracker {
	let started = false;
	let settled = false;
	let unsubscribe: (() => void) | null = null;

	function release() {
		unsubscribe?.();
		unsubscribe = null;
	}

	unsubscribe = onRunStarted((handle) => {
		if (started) return;
		started = true;
		release();
		void handle.done.then((outcome) => {
			if (outcome.status !== 'completed' && !settled) onAbandon();
		});
	});

	return {
		delivered() {
			settled = true;
			release();
		},
		closed() {
			if (started || settled) return;
			release();
			onAbandon();
		}
	};
}
