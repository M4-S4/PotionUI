export const VIDEO_HANDOFF_TIMEOUT_MS = 8000;

export function createPreviewKeeper() {
	let generationId: string | null | undefined;
	let held: string | null = null;
	return (id: string | null | undefined, image: string | null | undefined): string | null => {
		if (id !== generationId) {
			generationId = id;
			held = null;
		}
		if (image && image.startsWith('data:')) held = image;
		return held;
	};
}

export function createVideoHandoff(
	setPending: (pending: boolean) => void,
	timeoutMs: number = VIDEO_HANDOFF_TIMEOUT_MS
) {
	let settled: string | null | undefined;
	let waiting: string | null = null;
	let timer: ReturnType<typeof setTimeout> | null = null;

	function clearTimer() {
		if (timer !== null) clearTimeout(timer);
		timer = null;
	}

	function ready() {
		if (waiting !== null) settled = waiting;
		waiting = null;
		clearTimer();
		setPending(false);
	}

	function want(wanted: string | null | undefined, previewHeld: boolean) {
		if (!wanted) {
			clearTimer();
			waiting = null;
			setPending(false);
			return;
		}
		if (wanted === settled || wanted === waiting) return;
		clearTimer();
		waiting = null;
		if (!previewHeld) {
			settled = wanted;
			setPending(false);
			return;
		}
		waiting = wanted;
		setPending(true);
		timer = setTimeout(ready, timeoutMs);
	}

	return { want, ready };
}
