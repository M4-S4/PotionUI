import { createVisiblePoll } from '$lib/utils/visiblePoll';
import type { ModelDigestRequestResult } from '$lib/services/api/models';

const MIN_DELAY_MS = 3000;
const MAX_DELAY_MS = 30000;
const ETA_DIVISOR = 10;
const MAX_ATTEMPTS = 120;

export type DigestWatchFailure = { reason: 'request'; error?: unknown } | { reason: 'timeout' } | { reason: 'failed' };

export interface DigestWatchOptions {
	request: (retry: boolean) => Promise<ModelDigestRequestResult | null>;
	onProgress: (etaSeconds: number | null) => void;
	onReady: (mergedInto: string | null) => void;
	onFailed: (failure: DigestWatchFailure) => void;
	maxAttempts?: number;
}

export interface DigestWatch {
	start(options?: { retry?: boolean }): void;
	stop(): void;
}

export function digestWatchDelay(etaSeconds: number | null): number {
	if (etaSeconds == null || !Number.isFinite(etaSeconds)) return MIN_DELAY_MS;
	return Math.min(MAX_DELAY_MS, Math.max(MIN_DELAY_MS, Math.round((etaSeconds * 1000) / ETA_DIVISOR)));
}

export function createDigestWatch(options: DigestWatchOptions): DigestWatch {
	const maxAttempts = options.maxAttempts ?? MAX_ATTEMPTS;
	let delay = MIN_DELAY_MS;
	let attempts = 0;
	let generation = 0;
	let running = false;
	let retryNext = false;
	const poll = createVisiblePoll(step, () => delay);

	function finish() {
		running = false;
		poll.stop();
	}

	async function step() {
		const mine = generation;
		if (attempts >= maxAttempts) {
			finish();
			options.onFailed({ reason: 'timeout' });
			return;
		}
		attempts += 1;
		const retry = retryNext;
		retryNext = false;
		let result: ModelDigestRequestResult | null = null;
		let error: unknown;
		try {
			result = await options.request(retry);
		} catch (e) {
			error = e;
		}
		if (!running || mine !== generation) return;
		if (!result) {
			finish();
			options.onFailed({ reason: 'request', error });
			return;
		}
		if (result.state === 'ready') {
			finish();
			options.onReady(result.merged_into ?? null);
			return;
		}
		if (result.state === 'failed') {
			finish();
			options.onFailed({ reason: 'failed' });
			return;
		}
		delay = digestWatchDelay(result.eta_seconds);
		options.onProgress(result.eta_seconds);
	}

	return {
		start(startOptions) {
			poll.stop();
			generation += 1;
			const mine = generation;
			running = true;
			retryNext = !!startOptions?.retry;
			attempts = 0;
			delay = MIN_DELAY_MS;
			void step().then(() => {
				if (running && mine === generation) poll.start();
			});
		},
		stop() {
			running = false;
			generation += 1;
			poll.stop();
		}
	};
}
