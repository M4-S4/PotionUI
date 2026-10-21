export interface KeyValueRow {
	name: string;
	value: string;
	secret?: boolean;
}

export function normalizeRows(value: unknown): KeyValueRow[] {
	if (!Array.isArray(value)) return [];
	return value
		.filter((row): row is Record<string, unknown> => !!row && typeof row === 'object')
		.map((row) => ({
			name: String(row.name ?? ''),
			value: String(row.value ?? ''),
			...(row.secret ? { secret: true } : {})
		}));
}

export function addRow(rows: KeyValueRow[]): KeyValueRow[] {
	return [...rows, { name: '', value: '' }];
}

export function removeRow(rows: KeyValueRow[], index: number): KeyValueRow[] {
	return rows.filter((_, i) => i !== index);
}

export function patchRow(rows: KeyValueRow[], index: number, patch: Partial<KeyValueRow>): KeyValueRow[] {
	return rows.map((row, i) => {
		if (i !== index) return row;
		const next = { ...row, ...patch };
		if (!next.secret) delete next.secret;
		return next;
	});
}
