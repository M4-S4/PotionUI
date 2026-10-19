export interface ParsedApiError {
	message: string | undefined;
	fieldErrors: Record<string, string>;
}

const VALUE_ERROR_PREFIX = /^value error,\s*/i;

function readableValidationMessage(raw: string): string {
	const text = raw.replace(VALUE_ERROR_PREFIX, '').trim();
	return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

function fieldOf(loc: unknown): string | undefined {
	if (!Array.isArray(loc)) return undefined;
	const names = loc.filter((part): part is string => typeof part === 'string' && part !== 'body');
	return names[names.length - 1];
}

export function parseApiError(error: any): ParsedApiError {
	const data = error?.response?.data;
	const detail = data?.detail;
	const fieldErrors: Record<string, string> = {};

	if (Array.isArray(detail)) {
		for (const item of detail) {
			const field = fieldOf(item?.loc);
			if (field && typeof item?.msg === 'string' && !(field in fieldErrors)) {
				fieldErrors[field] = readableValidationMessage(item.msg);
			}
		}
		const first = Object.values(fieldErrors)[0];
		if (first) {
			return { message: first, fieldErrors };
		}
		const summary = typeof data?.message === 'string' ? data.message : undefined;
		return { message: summary ?? error?.message, fieldErrors };
	}

	if (typeof detail === 'string') return { message: detail, fieldErrors };
	if (detail && typeof detail.message === 'string') return { message: detail.message, fieldErrors };
	if (typeof data?.message === 'string') return { message: data.message, fieldErrors };
	return { message: error?.message, fieldErrors };
}
