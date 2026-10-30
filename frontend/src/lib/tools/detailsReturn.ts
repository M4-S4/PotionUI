import type { LandingSource } from '$lib/library/landing';

type Handler = (target: LandingSource) => void;

let handler: Handler | null = null;

export function registerDetailsReturn(next: Handler): () => void {
	handler = next;
	return () => {
		if (handler === next) handler = null;
	};
}

export function canReturnToDetails(): boolean {
	return handler !== null;
}

export function returnToDetails(target: LandingSource): boolean {
	if (!handler) return false;
	handler(target);
	return true;
}
