/**
 * Pure utility functions shared across AI chat panel components.
 */

import { parseServerDate } from '$lib/utils/relativeTime';
import { startOfDay } from '$lib/utils/time';

/**
 * Format a token count with a 'k' suffix for values >= 1000.
 */
export function formatTokenCount(n: number): string {
	if (n >= 1000) {
		return (n / 1000).toFixed(1) + 'k';
	}
	return n.toString();
}

export interface SessionDateGroup<T> {
	label: 'Today' | 'Yesterday' | 'This week' | 'Older';
	sessions: T[];
}

/**
 * Bucket chat sessions for the history view by their last activity.
 * Preserves the input order inside each bucket; empty buckets are omitted.
 */
export function groupSessionsByDate<T extends { updated_at?: string; created_at?: string }>(
	sessions: T[],
	now: Date = new Date()
): SessionDateGroup<T>[] {
	const today = startOfDay(now);

	const buckets: Record<SessionDateGroup<T>['label'], T[]> = {
		Today: [],
		Yesterday: [],
		'This week': [],
		Older: []
	};

	for (const session of sessions) {
		const raw = session.updated_at || session.created_at;
		const date = raw ? parseServerDate(raw) : null;
		if (!date) {
			buckets.Older.push(session);
			continue;
		}
		const dayDiff = Math.round((today - startOfDay(date)) / 86_400_000);
		if (dayDiff <= 0) buckets.Today.push(session);
		else if (dayDiff === 1) buckets.Yesterday.push(session);
		else if (dayDiff < 7) buckets['This week'].push(session);
		else buckets.Older.push(session);
	}

	return (Object.keys(buckets) as SessionDateGroup<T>['label'][])
		.filter((label) => buckets[label].length > 0)
		.map((label) => ({ label, sessions: buckets[label] }));
}
