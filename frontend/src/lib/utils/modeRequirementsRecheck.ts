export const MODE_RECHECK_RETRY_MS = 2000;

export interface ModeRecheckTarget {
	tabId: string;
	presetId: string;
	mode: string;
}

export interface ModeRecheckDeps {
	target(): ModeRecheckTarget | null;
	isBlocked(target: ModeRecheckTarget): boolean;
	reload(target: ModeRecheckTarget): Promise<void>;
	retryMs?: number;
}

function sameTarget(a: ModeRecheckTarget, b: ModeRecheckTarget | null): b is ModeRecheckTarget {
	return !!b && a.tabId === b.tabId && a.presetId === b.presetId && a.mode === b.mode;
}

export function createModeRequirementsRecheck(deps: ModeRecheckDeps) {
	const retryMs = deps.retryMs ?? MODE_RECHECK_RETRY_MS;
	let timer: ReturnType<typeof setTimeout> | null = null;
	let latestRun = 0;
	let disposed = false;

	function cancelRetry() {
		if (timer !== null) {
			clearTimeout(timer);
			timer = null;
		}
	}

	async function pass(): Promise<boolean> {
		const target = deps.target();
		if (!target || !deps.isBlocked(target)) return false;
		await deps.reload(target);
		if (disposed) return false;
		const after = deps.target();
		return sameTarget(target, after) && deps.isBlocked(after);
	}

	async function run(): Promise<void> {
		if (disposed) return;
		cancelRetry();
		const thisRun = ++latestRun;
		const stillBlocked = await pass();
		if (!stillBlocked || disposed || thisRun !== latestRun) return;
		timer = setTimeout(() => {
			timer = null;
			void pass();
		}, retryMs);
	}

	function dispose() {
		disposed = true;
		cancelRetry();
	}

	return { run, dispose };
}
