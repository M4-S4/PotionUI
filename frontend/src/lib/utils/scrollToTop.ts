function isVerticalScroller(el: HTMLElement): boolean {
	const overflowY = getComputedStyle(el).overflowY;
	return overflowY === 'auto' || overflowY === 'scroll';
}

export function scrollToTop(from: HTMLElement | null): void {
	for (let el = from?.parentElement ?? null; el; el = el.parentElement) {
		if (isVerticalScroller(el)) el.scrollTop = 0;
	}
	window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
}

export function resetScrollOnChange<T>(node: HTMLElement, value: T) {
	let shown = value;
	return {
		update(next: T) {
			if (next === shown) return;
			shown = next;
			scrollToTop(node);
		}
	};
}
