export const TERMINAL_TYPES: ReadonlySet<string> = new Set([
	'generation_complete',
	'generation_error',
	'generation_cancelled'
]);

interface WireFrame {
	type: string;
	generation_id?: string;
	data?: unknown;
}

function text(value: unknown): string | undefined {
	return typeof value === 'string' && value !== '' ? value : undefined;
}

export function generationIdOf(message: WireFrame): string | undefined {
	if (TERMINAL_TYPES.has(message.type)) {
		const data = message.data as { id?: unknown; generation_id?: unknown } | null | undefined;
		return text(data?.id) ?? text(data?.generation_id) ?? message.generation_id;
	}
	return message.generation_id;
}
