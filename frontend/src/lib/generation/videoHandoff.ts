export type VideoPreloader = (url: string, onSettled: () => void) => void;

export const VIDEO_HANDOFF_TIMEOUT_MS = 8000;

export function preloadVideo(url: string, onSettled: () => void): void {
	if (typeof document === 'undefined') {
		onSettled();
		return;
	}
	let done = false;
	const settle = () => {
		if (done) return;
		done = true;
		clearTimeout(timer);
		onSettled();
	};
	const probe = document.createElement('video');
	probe.preload = 'auto';
	probe.muted = true;
	probe.onloadeddata = settle;
	probe.onerror = settle;
	const timer = setTimeout(settle, VIDEO_HANDOFF_TIMEOUT_MS);
	probe.src = url;
}

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
	preload: VideoPreloader = preloadVideo,
	resolveUrl: (url: string) => string = (url) => url
) {
	let settled: string | null | undefined;
	let inflight: string | null = null;
	let latest = 0;
	return (wanted: string | null | undefined, previewHeld: boolean): void => {
		if (wanted && wanted === inflight) return;
		const ticket = ++latest;
		inflight = null;
		if (!wanted || wanted === settled || !previewHeld) {
			settled = wanted;
			setPending(false);
			return;
		}
		inflight = wanted;
		setPending(true);
		preload(resolveUrl(wanted), () => {
			if (ticket !== latest) return;
			settled = wanted;
			inflight = null;
			setPending(false);
		});
	};
}
