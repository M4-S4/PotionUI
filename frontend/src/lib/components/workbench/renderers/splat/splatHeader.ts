export interface SourceView {
	fovY: number;
	aspect: number | null;
	depth: number;
	width: number | null;
	height: number | null;
}

export const MAX_HEADER_BYTES = 65536;

const VIEW_COMMENT = 'comment potionui_view';
const HEADER_END = /(^|\n)end_header\r?\n/;

function isPly(bytes: Uint8Array): boolean {
	return bytes.length >= 4 && bytes[0] === 0x70 && bytes[1] === 0x6c && bytes[2] === 0x79;
}

export function readPlyHeader(bytes: Uint8Array, limit = MAX_HEADER_BYTES): string | null {
	if (!isPly(bytes)) return null;
	const text = new TextDecoder('latin1').decode(bytes.subarray(0, Math.min(bytes.length, limit)));
	const end = HEADER_END.exec(text);
	return end ? text.slice(0, end.index + end[0].length) : null;
}

function numberField(value: string | undefined): number | null {
	if (value === undefined || value === '') return null;
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : null;
}

export function parseSourceView(bytes: Uint8Array): SourceView | null {
	const header = readPlyHeader(bytes);
	if (!header) return null;

	for (const line of header.split(/\r?\n/)) {
		if (line !== VIEW_COMMENT && !line.startsWith(`${VIEW_COMMENT} `)) continue;

		const fields = new Map<string, string>();
		for (const token of line.slice(VIEW_COMMENT.length).trim().split(/\s+/)) {
			const equals = token.indexOf('=');
			if (equals > 0) fields.set(token.slice(0, equals), token.slice(equals + 1));
		}

		const fovY = numberField(fields.get('fov_y'));
		const depth = numberField(fields.get('depth'));
		if (fovY === null || fovY <= 0 || fovY >= 180) return null;
		if (depth === null || depth <= 0) return null;
		if (fields.has('convention') && fields.get('convention') !== 'opencv') return null;

		return {
			fovY,
			depth,
			aspect: numberField(fields.get('aspect')),
			width: numberField(fields.get('width')),
			height: numberField(fields.get('height'))
		};
	}
	return null;
}
