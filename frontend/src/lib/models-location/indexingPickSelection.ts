import type { IndexingDigestRequest, IndexingPendingItem } from '$lib/services/api/models';

export interface PickFilter {
	query: string;
	modelType: string | null;
}

type PickedFile = Pick<IndexingPendingItem, 'model_id' | 'filename' | 'model_type' | 'size'>;

interface MatchingScope extends PickFilter {
	count: number;
	bytes: number;
	excluded: Map<string, PickedFile>;
}

export interface PickSelection {
	ticked: Map<string, PickedFile>;
	scopes: MatchingScope[];
}

export function emptyPickSelection(): PickSelection {
	return { ticked: new Map(), scopes: [] };
}

function normalizedQuery(query: string): string {
	return query.trim().toLowerCase();
}

function matchesFilter(filter: PickFilter, file: Pick<PickedFile, 'filename' | 'model_type'>): boolean {
	if (filter.modelType && file.model_type !== filter.modelType) return false;
	const query = normalizedQuery(filter.query);
	return !query || file.filename.toLowerCase().includes(query);
}

function covers(outer: PickFilter, inner: PickFilter): boolean {
	if (outer.modelType && outer.modelType !== inner.modelType) return false;
	return normalizedQuery(inner.query).includes(normalizedQuery(outer.query));
}

function disjoint(a: PickFilter, b: PickFilter): boolean {
	return !!a.modelType && !!b.modelType && a.modelType !== b.modelType;
}

function pickedFile(item: IndexingPendingItem): PickedFile {
	return { model_id: item.model_id, filename: item.filename, model_type: item.model_type, size: item.size };
}

export function pickIsSelected(selection: PickSelection, item: IndexingPendingItem): boolean {
	if (selection.ticked.has(item.model_id)) return true;
	return selection.scopes.some((scope) => matchesFilter(scope, item) && !scope.excluded.has(item.model_id));
}

export function pickToggle(selection: PickSelection, item: IndexingPendingItem, selected: boolean): PickSelection {
	const ticked = new Map(selection.ticked);
	const scopes = selection.scopes.map((scope) => ({ ...scope, excluded: new Map(scope.excluded) }));
	const matching = scopes.filter((scope) => matchesFilter(scope, item));
	ticked.delete(item.model_id);
	for (const scope of matching) {
		if (selected) scope.excluded.delete(item.model_id);
		else scope.excluded.set(item.model_id, pickedFile(item));
	}
	if (selected && matching.length === 0) ticked.set(item.model_id, pickedFile(item));
	return { ticked, scopes };
}

export function pickSelectAllMatching(
	selection: PickSelection,
	filter: PickFilter,
	count: number,
	bytes: number
): PickSelection {
	const incoming: MatchingScope = { ...filter, count, bytes, excluded: new Map() };
	const scopes: MatchingScope[] = [];
	let alreadyCovered = false;
	for (const scope of selection.scopes) {
		if (covers(incoming, scope)) continue;
		if (covers(scope, incoming)) {
			alreadyCovered = true;
			const excluded = new Map([...scope.excluded].filter(([, file]) => !matchesFilter(filter, file)));
			scopes.push({ ...scope, excluded });
		} else {
			scopes.push(scope);
		}
	}
	if (!alreadyCovered) scopes.push(incoming);
	const ticked = new Map<string, PickedFile>();
	for (const [id, file] of selection.ticked) {
		if (!matchesFilter(filter, file)) ticked.set(id, file);
	}
	return { ticked, scopes };
}

export function pickCount(selection: PickSelection): number {
	let count = selection.ticked.size;
	for (const scope of selection.scopes) count += Math.max(0, scope.count - scope.excluded.size);
	return count;
}

export function pickBytes(selection: PickSelection): number {
	let bytes = 0;
	for (const file of selection.ticked.values()) bytes += file.size;
	for (const scope of selection.scopes) {
		let excluded = 0;
		for (const file of scope.excluded.values()) excluded += file.size;
		bytes += Math.max(0, scope.bytes - excluded);
	}
	return bytes;
}

export function pickIsApproximate(selection: PickSelection): boolean {
	const { scopes } = selection;
	for (let i = 0; i < scopes.length; i++) {
		for (let j = i + 1; j < scopes.length; j++) {
			if (!disjoint(scopes[i], scopes[j]) && !covers(scopes[i], scopes[j]) && !covers(scopes[j], scopes[i])) return true;
		}
	}
	return false;
}

export function pickRequests(selection: PickSelection): IndexingDigestRequest[] {
	const requests: IndexingDigestRequest[] = [];
	for (const scope of selection.scopes) {
		const request: IndexingDigestRequest = { mode: 'matching' };
		const query = scope.query.trim();
		if (query) request.query = query;
		if (scope.modelType) request.model_type = scope.modelType;
		if (scope.excluded.size) request.exclude_model_ids = [...scope.excluded.keys()];
		requests.push(request);
	}
	if (selection.ticked.size) requests.push({ mode: 'selected', model_ids: [...selection.ticked.keys()] });
	return requests;
}

export function pickAllMatchingSelected(selection: PickSelection, filter: PickFilter, total: number): boolean {
	if (total === 0) return false;
	return selection.scopes.some(
		(scope) => covers(scope, filter) && ![...scope.excluded.values()].some((file) => matchesFilter(filter, file))
	);
}
