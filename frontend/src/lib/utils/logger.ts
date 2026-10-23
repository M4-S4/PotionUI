import { parseApiError } from '$lib/utils/apiErrors';

const isDev = typeof import.meta !== 'undefined' && import.meta.env?.DEV;

export const logger = {
	debug: (...args: unknown[]) => {
		if (isDev) console.log(...args);
	},
	warn: (...args: unknown[]) => {
		if (isDev) console.warn(...args);
	},
	error: (...args: unknown[]) => console.error(...args)
};

/**
 * Extracts a human-readable message from an unknown caught error.
 * Falls back to the provided default message if the error has no message.
 */
export function getErrorMessage(err: unknown, fallback = 'An unexpected error occurred'): string {
	if (err instanceof Error) return err.message;
	if (typeof err === 'string') return err;
	if (
		err !== null &&
		typeof err === 'object' &&
		'message' in err &&
		typeof (err as Record<string, unknown>).message === 'string'
	) {
		return (err as Record<string, unknown>).message as string;
	}
	return fallback;
}

export function getApiErrorMessage(err: unknown, fallback = 'An unexpected error occurred'): string {
	return parseApiError(err).message || getErrorMessage(err, fallback);
}
