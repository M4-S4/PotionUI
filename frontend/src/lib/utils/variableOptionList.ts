export const OPTION_FILTER_THRESHOLD = 6;

export function filterOptionIndices(texts: string[], query: string): number[] {
	const needle = query.trim().toLowerCase();
	const indices: number[] = [];
	texts.forEach((text, index) => {
		if (!needle || text.toLowerCase().includes(needle)) indices.push(index);
	});
	return indices;
}

export function showOptionFilter(count: number): boolean {
	return count > OPTION_FILTER_THRESHOLD;
}
