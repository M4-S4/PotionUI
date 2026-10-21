import { parseServerDate } from '$lib/utils/relativeTime';

export function formatDate(dateString?: string | null): string {
	if (!dateString) return 'N/A';
	return parseServerDate(dateString)?.toLocaleString() ?? 'N/A';
}
