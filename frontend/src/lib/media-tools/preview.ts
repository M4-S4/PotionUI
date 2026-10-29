import { logger, getErrorMessage } from '$lib/utils/logger';

export const PREVIEW_DEBOUNCE_MS = 300;

export interface PreviewState {
	url: string | null;
	loading: boolean;
	error: string | null;
}

export interface PreviewControllerOptions {
	request: (signal: AbortSignal) => Promise<Blob>;
	onChange: (state: PreviewState) => void;
	debounceMs?: number;
	createUrl?: (blob: Blob) => string;
	revokeUrl?: (url: string) => void;
}

export interface PreviewController {
	schedule(immediate?: boolean): void;
	cancel(): void;
	clear(): void;
	destroy(): void;
}

function isAbort(error: unknown): boolean {
	const name = (error as { name?: string } | null)?.name;
	const code = (error as { code?: string } | null)?.code;
	return name === 'AbortError' || name === 'CanceledError' || code === 'ERR_CANCELED';
}

export function createPreviewController(options: PreviewControllerOptions): PreviewController {
	const debounceMs = options.debounceMs ?? PREVIEW_DEBOUNCE_MS;
	const createUrl = options.createUrl ?? ((blob: Blob) => URL.createObjectURL(blob));
	const revokeUrl = options.revokeUrl ?? ((url: string) => URL.revokeObjectURL(url));

	let timer: ReturnType<typeof setTimeout> | null = null;
	let controller: AbortController | null = null;
	let url: string | null = null;
	let destroyed = false;

	function emit(loading: boolean, error: string | null) {
		if (!destroyed) options.onChange({ url, loading, error });
	}

	function abort() {
		if (timer !== null) clearTimeout(timer);
		timer = null;
		controller?.abort();
		controller = null;
	}

	function dropUrl() {
		if (url) revokeUrl(url);
		url = null;
	}

	async function run() {
		timer = null;
		const mine = new AbortController();
		controller = mine;
		try {
			const blob = await options.request(mine.signal);
			if (mine.signal.aborted || destroyed) return;
			const next = createUrl(blob);
			dropUrl();
			url = next;
			controller = null;
			emit(false, null);
		} catch (error) {
			if (mine.signal.aborted || destroyed || isAbort(error)) return;
			controller = null;
			logger.error('Media tool preview failed:', getErrorMessage(error));
			emit(false, 'The preview could not be rendered');
		}
	}

	return {
		schedule(immediate = false) {
			if (destroyed) return;
			abort();
			emit(true, null);
			if (immediate) void run();
			else timer = setTimeout(() => void run(), debounceMs);
		},
		cancel() {
			abort();
			emit(false, null);
		},
		clear() {
			abort();
			dropUrl();
			emit(false, null);
		},
		destroy() {
			abort();
			dropUrl();
			destroyed = true;
		}
	};
}
