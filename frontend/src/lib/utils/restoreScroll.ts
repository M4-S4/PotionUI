const GIVE_UP_MS = 2000;

export function restoreScroll(node: HTMLElement, top: number | null) {
	if (top === null || top <= 0) return {};
	let done = false;
	let observer: ResizeObserver | null = null;
	const interrupts = ['wheel', 'touchstart', 'keydown', 'pointerdown'] as const;

	function stop() {
		if (done) return;
		done = true;
		observer?.disconnect();
		clearTimeout(timer);
		for (const name of interrupts) node.removeEventListener(name, stop);
	}

	function apply() {
		if (done) return;
		node.scrollTop = top as number;
		if (node.scrollTop >= (top as number) - 1) stop();
	}

	const timer = setTimeout(stop, GIVE_UP_MS);
	for (const name of interrupts) node.addEventListener(name, stop, { passive: true });
	if (typeof ResizeObserver !== 'undefined' && node.firstElementChild) {
		observer = new ResizeObserver(apply);
		observer.observe(node.firstElementChild);
	}
	apply();
	return { destroy: stop };
}
