export type ImagePreloader = (url: string, onSettled: () => void) => void;

export function preloadImage(url: string, onSettled: () => void): void {
	if (typeof Image === 'undefined') {
		onSettled();
		return;
	}
	const probe = new Image();
	probe.onload = onSettled;
	probe.onerror = onSettled;
	probe.src = url;
}

const isInlinePreview = (url: string | null | undefined): boolean => !!url && url.startsWith('data:');

export function createImageHandoff(
	show: (url: string | null | undefined) => void,
	preload: ImagePreloader = preloadImage,
	resolveUrl: (url: string) => string = (url) => url
) {
	let shown: string | null | undefined;
	let latest = 0;
	return (wanted: string | null | undefined): void => {
		const ticket = ++latest;
		if (!(wanted && isInlinePreview(shown) && !isInlinePreview(wanted))) {
			shown = wanted;
			show(wanted);
			return;
		}
		preload(resolveUrl(wanted), () => {
			if (ticket !== latest) return;
			shown = wanted;
			show(wanted);
		});
	};
}
