import type { Session, SessionVersionSummary } from '$lib/types/api';
import { parseServerDate } from '$lib/utils/relativeTime';

export function filterSessions(sessions: Session[], query: string): Session[] {
	const needle = query.trim().toLowerCase();
	return sessions.filter((session) => !needle || session.name.toLowerCase().includes(needle));
}

export function splitPinned(sessions: Session[]): { pinned: Session[]; rest: Session[] } {
	return {
		pinned: sessions.filter((session) => session.pinned),
		rest: sessions.filter((session) => !session.pinned)
	};
}

function updatedTime(session: Session): number {
	return parseServerDate(session.updated_at)?.getTime() ?? 0;
}

export function applyPin(sessions: Session[], sessionId: string, pinned: boolean): Session[] {
	const target = sessions.find((session) => session.id === sessionId);
	if (!target) return sessions;
	const others = sessions.filter((session) => session.id !== sessionId);
	const updated = { ...target, pinned };
	const keptPinned = others.filter((session) => session.pinned);
	const rest = others.filter((session) => !session.pinned);
	if (pinned) return [updated, ...keptPinned, ...rest];
	return [...keptPinned, ...[...rest, updated].sort((a, b) => updatedTime(b) - updatedTime(a))];
}

export function newestFirst(sessions: Session[]): Session[] {
	return [...sessions].sort((a, b) => updatedTime(b) - updatedTime(a));
}

export function versionHeadline(version: SessionVersionSummary): string {
	return version.prompt_preview?.trim() || version.summary;
}

export const CHANGE_LABELS_SHOWN = 3;

export type FieldLabels = Record<string, string>;

export interface VersionChangeSummary {
	shown: string[];
	more: number;
	all: string[];
}

export function humanizeKey(key: string): string {
	return key.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
}

export function versionChanges(
	version: SessionVersionSummary,
	fieldLabels: FieldLabels = {}
): VersionChangeSummary | null {
	const labels = [...(version.changes ?? [])];
	for (const key of version.changed_fields ?? []) {
		const label = fieldLabels[key] || humanizeKey(key);
		if (label && !labels.includes(label)) labels.push(label);
	}
	if (labels.length === 0) return null;
	return {
		shown: labels.slice(0, CHANGE_LABELS_SHOWN),
		more: Math.max(0, labels.length - CHANGE_LABELS_SHOWN),
		all: labels
	};
}

export type DiscardAction =
	| { kind: 'load' }
	| { kind: 'restore'; versionNumber: number }
	| { kind: 'new' };

export function discardConsequence(action: DiscardAction): string {
	if (action.kind === 'restore') return `Restoring version v${action.versionNumber} replaces them.`;
	if (action.kind === 'new') return 'Starting a new session clears them.';
	return 'Loading another session replaces them.';
}

export function fullDateTime(iso: string | undefined): string {
	const date = parseServerDate(iso);
	return date ? date.toLocaleString() : '';
}
