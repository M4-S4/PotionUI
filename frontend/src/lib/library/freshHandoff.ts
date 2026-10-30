let pending: string[] = [];

export function setPendingFresh(ids: readonly string[]): void {
	pending = [...ids];
}

export function takePendingFresh(): string[] {
	const ids = pending;
	pending = [];
	return ids;
}

export function removePendingFresh(id: string): void {
	pending = pending.filter((entry) => entry !== id);
}
