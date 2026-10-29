export const FILTERS_TOOL = 'filters';

export function lookTool(lookId: string): string {
	return `${FILTERS_TOOL}:${lookId}`;
}

export function parseInitialTool(value: string): { tool: string; look: string | null } {
	const at = value.indexOf(':');
	if (at === -1) return { tool: value, look: null };
	const look = value.slice(at + 1);
	return { tool: value.slice(0, at), look: look || null };
}
