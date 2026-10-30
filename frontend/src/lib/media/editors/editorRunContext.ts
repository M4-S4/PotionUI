import type { Readable } from 'svelte/store';

export const EDITOR_RUN_CONTEXT_KEY = 'media-editor-run';

export interface EditorRunContext {
	progress: Readable<number | null>;
	cancelling: Readable<boolean>;
	cancel: () => void;
}
