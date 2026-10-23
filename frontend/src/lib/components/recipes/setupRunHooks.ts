import type { SetupRun } from '$lib/services/api/setup';

export const SETUP_POLL_RETRY_NOTICE = "Couldn't check on setup progress — trying again shortly.";

export interface SetupRunHooksTarget {
	setRun(run: SetupRun | null): void;
	setFetchError(message: string): void;
	remember(runId: string): void;
	forget(): void;
}

export function createSetupRunHooks(target: SetupRunHooksTarget) {
	return {
		onRun(run: SetupRun | null): void {
			target.setRun(run);
			if (run) {
				target.setFetchError('');
				target.remember(run.id);
			} else {
				target.forget();
			}
		},
		onPollError(error: unknown): 'drop' | 'retry' {
			if ((error as { response?: { status?: number } } | null)?.response?.status === 404) return 'drop';
			target.setFetchError(SETUP_POLL_RETRY_NOTICE);
			return 'retry';
		}
	};
}
