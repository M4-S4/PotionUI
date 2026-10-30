import { dayKey, dayLabel } from '$lib/utils/relativeTime';

export interface DayGroup<T> {
	key: string;
	label: string;
	items: T[];
}

interface Dated {
	id: string;
	created_at?: string;
}

function ownKey(item: Dated): string {
	return item.created_at ? dayKey(item.created_at) : '__undated__';
}

function ownLabel(item: Dated): string {
	return item.created_at ? dayLabel(item.created_at) : 'Undated';
}

export function groupByDay<T extends Dated>(list: T[], freshIds: readonly string[] = []): DayGroup<T>[] {
	const fresh = new Set(freshIds);
	const anchors: T[] = list.map((item, index) => {
		if (!fresh.has(item.id)) return item;
		for (let next = index + 1; next < list.length; next += 1) {
			if (!fresh.has(list[next].id)) return list[next];
		}
		return item;
	});

	const groups: DayGroup<T>[] = [];
	const seen = new Map<string, number>();
	list.forEach((item, index) => {
		const anchor = anchors[index];
		const key = ownKey(anchor);
		const last = groups[groups.length - 1];
		if (last && last.key.split('#')[0] === key) {
			last.items.push(item);
			return;
		}
		const occurrence = seen.get(key) ?? 0;
		seen.set(key, occurrence + 1);
		groups.push({
			key: occurrence === 0 ? key : `${key}#${occurrence}`,
			label: ownLabel(anchor),
			items: [item]
		});
	});
	return groups;
}
