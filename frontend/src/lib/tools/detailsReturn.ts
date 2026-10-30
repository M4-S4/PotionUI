import type { LandingSource } from '$lib/library/landing';

type Handler = (target: LandingSource) => void;

let handler: Handler | null = null;
let returningTo: string | null = null;

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
	returningTo = target.generationId;
	handler(target);
	return true;
}

const scrollTops = new Map<string, number>();

export function rememberDetailsScroll(generationId: string, top: number): void {
	if (generationId && top > 0) scrollTops.set(generationId, top);
	else scrollTops.delete(generationId);
}

export function takeDetailsScroll(generationId: string): number | null {
	if (returningTo !== generationId) return null;
	returningTo = null;
	const top = scrollTops.get(generationId);
	scrollTops.delete(generationId);
	return top ?? null;
}
