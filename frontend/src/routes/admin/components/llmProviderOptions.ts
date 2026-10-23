import type { LLMContextWindowDiscovery } from '$lib/types/llm';

/**
 * A text-typed provider option stays text unless the WHOLE value is a numeric
 * literal. Ollama durations ("5m", "1h") begin with digits, so `parseFloat`
 * silently sends 5 and 1.
 */
export function coerceProviderOptionText(raw: string): string | number {
	const trimmed = raw.trim();
	if (trimmed === '') return raw;
	const asNumber = Number(trimmed);
	return Number.isFinite(asNumber) ? asNumber : raw;
}

export function migrateLegacyContextWindow(options: Record<string, any>): Record<string, any> {
	if (!('num_ctx' in options)) return options;
	const { num_ctx, ...rest } = options;
	if (rest.context_window === undefined || rest.context_window === null) {
		if (typeof num_ctx === 'number' && Number.isFinite(num_ctx)) rest.context_window = num_ctx;
	}
	return rest;
}

export type ContextReadoutPart = string | { n: number };

export interface ContextReadout {
	tone: 'info' | 'warning';
	parts: ContextReadoutPart[];
}

export const CONTEXT_FALLBACK_TOKENS = 8192;

export function contextWindowReadout(
	result: LLMContextWindowDiscovery | null,
	fieldSet: boolean,
	requestError: string | null
): ContextReadout | null {
	const error = requestError ?? result?.error ?? null;
	if (error) {
		return fieldSet
			? { tone: 'warning', parts: [error] }
			: { tone: 'warning', parts: [`${error} Using `, { n: CONTEXT_FALLBACK_TOKENS }, ' tokens until a window is set.'] };
	}
	if (!result) return null;
	if (!result.supported) {
		return fieldSet ? null : { tone: 'info', parts: ["This provider can't report its window — set it here"] };
	}
	if (fieldSet) {
		return result.detected != null ? { tone: 'info', parts: ['Detected ', { n: result.detected }, ' tokens'] } : null;
	}
	if (result.effective == null) return null;
	if (result.detected_kind === 'trained' && result.detected != null && result.effective < result.detected) {
		return {
			tone: 'info',
			parts: ['Using ', { n: result.effective }, ' tokens. The model supports ', { n: result.detected }, '; raise the window if your GPU has room']
		};
	}
	return {
		tone: 'info',
		parts: ['Using ', { n: result.effective }, result.detected_kind === 'trained' ? ' tokens, from the model' : ' tokens, detected from the server']
	};
}
