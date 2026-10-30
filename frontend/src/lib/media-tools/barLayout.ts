export const BAR_GAP = 4;
export const BAR_DIVIDER = 7;
export const BAR_SPACER = 8;

export interface BarLayoutInput {
	available: number;
	widths: Readonly<Record<string, number>>;
	favorite: boolean;
	tags: boolean;
	pageIds: readonly string[];
	download: boolean;
	effects: boolean;
	edit: boolean;
}

export function foldOrder(input: Pick<BarLayoutInput, 'favorite' | 'tags' | 'pageIds' | 'download'>): string[] {
	return [
		...(input.download ? ['download'] : []),
		...[...input.pageIds].reverse(),
		...(input.tags ? ['tags'] : []),
		...(input.favorite ? ['favorite'] : [])
	];
}

function sequence(input: BarLayoutInput, folded: ReadonlySet<string>): number[] {
	const widths = input.widths;
	const width = (id: string) => widths[id] ?? 0;
	const left = [...(input.edit ? ['edit'] : []), ...(input.effects ? ['effects'] : [])];
	const middle = [
		...(input.favorite ? ['favorite'] : []),
		...(input.tags ? ['tags'] : [])
	].filter((id) => !folded.has(id));
	const page = input.pageIds.filter((id) => !folded.has(id));
	const right = [...(input.download && !folded.has('download') ? ['download'] : []), 'more'];

	const out: number[] = left.map(width);
	if (middle.length > 0) out.push(BAR_DIVIDER, ...middle.map(width));
	if (page.length > 0) out.push(BAR_DIVIDER, ...page.map(width));
	out.push(BAR_SPACER, ...right.map(width));
	return out;
}

export function foldedIds(input: BarLayoutInput): Set<string> {
	const folded = new Set<string>();
	if (input.available <= 0) return folded;
	for (const id of foldOrder(input)) {
		const parts = sequence(input, folded);
		const total = parts.reduce((sum, part) => sum + part, 0) + BAR_GAP * (parts.length - 1);
		if (total <= input.available) break;
		folded.add(id);
	}
	return folded;
}
