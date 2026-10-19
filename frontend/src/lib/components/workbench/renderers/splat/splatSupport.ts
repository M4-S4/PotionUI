export const SPLAT_BACKGROUND = 0x1b1b1d;
export const SPLAT_BACKGROUND_CSS = '#1b1b1d';

export type SplatLoadErrorKind =
	| 'not-found'
	| 'forbidden'
	| 'http'
	| 'network'
	| 'decode'
	| 'empty'
	| 'timeout'
	| 'webgl';

export class SplatLoadError extends Error {
	readonly kind: SplatLoadErrorKind;
	readonly status: number | null;

	constructor(kind: SplatLoadErrorKind, status: number | null = null, cause?: unknown) {
		super(`splat load failed: ${kind}${status ? ` (${status})` : ''}`, { cause });
		this.name = 'SplatLoadError';
		this.kind = kind;
		this.status = status;
	}
}

export function splatErrorFromStatus(status: number): SplatLoadError {
	if (status === 404 || status === 410) return new SplatLoadError('not-found', status);
	if (status === 401 || status === 403) return new SplatLoadError('forbidden', status);
	return new SplatLoadError('http', status);
}

export function splatErrorMessage(error: unknown): string {
	if (!(error instanceof SplatLoadError)) {
		return 'This splat could not be displayed - the file may be corrupted or in an unsupported format.';
	}
	switch (error.kind) {
		case 'not-found':
			return 'This splat file could not be found - it may have been deleted.';
		case 'forbidden':
			return 'You do not have access to this splat file.';
		case 'http':
			return `The server could not deliver this splat file (HTTP ${error.status ?? 'error'}).`;
		case 'network':
			return 'This splat file could not be downloaded. Check your connection and try again.';
		case 'empty':
			return 'This file contains no splats.';
		case 'timeout':
			return 'The splat took too long to load.';
		case 'webgl':
			return 'WebGL is not available, so the splat viewer cannot start.';
		default:
			return 'This splat could not be displayed - the file may be corrupted or in an unsupported format.';
	}
}
