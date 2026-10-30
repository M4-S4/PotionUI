export const DISCONNECT_GRACE_MS = 4000;

export function createDisconnectBanner(
	show: (visible: boolean) => void,
	graceMs: number = DISCONNECT_GRACE_MS
) {
	let timer: ReturnType<typeof setTimeout> | null = null;

	function startGrace() {
		if (timer !== null) return;
		timer = setTimeout(() => {
			timer = null;
			show(true);
		}, graceMs);
	}

	function clearGrace() {
		if (timer !== null) clearTimeout(timer);
		timer = null;
	}

	return {
		connecting: startGrace,
		changed(connected: boolean) {
			if (connected) {
				clearGrace();
				show(false);
			} else {
				startGrace();
			}
		},
		destroy: clearGrace
	};
}
