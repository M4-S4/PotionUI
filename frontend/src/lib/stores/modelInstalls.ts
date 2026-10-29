import { logger } from '$lib/utils/logger';

type Listener = () => void;

const listeners = new Set<Listener>();

export function onModelInstalled(listener: Listener): () => void {
	listeners.add(listener);
	return () => {
		listeners.delete(listener);
	};
}

export function announceModelInstalled(): void {
	for (const listener of [...listeners]) {
		try {
			listener();
		} catch (error) {
			logger.error('[modelInstalls] Listener failed:', error);
		}
	}
}
