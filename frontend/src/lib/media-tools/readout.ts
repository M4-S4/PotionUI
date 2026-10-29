import type { MediaToolSource, ToolDTO } from './types';

export function formatReadoutTime(seconds: number): string {
	const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
	const tenths = Math.round(safe * 10);
	const whole = Math.floor(tenths / 10);
	const minutes = Math.floor(whole / 60);
	const secs = whole % 60;
	const clock = `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
	return `${clock}.${tenths % 10}`;
}

function num(value: unknown): number | null {
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function size(width: number | null | undefined, height: number | null | undefined): string | null {
	return width && height ? `${Math.round(width)}×${Math.round(height)}` : null;
}

function join(parts: Array<string | null>): string | null {
	const present = parts.filter((part): part is string => !!part);
	return present.length > 0 ? present.join(' · ') : null;
}

function evenFloor(value: number): number {
	return Math.max(2, Math.floor(value / 2) * 2);
}

function resizedSize(
	source: MediaToolSource,
	values: Record<string, unknown>
): { width: number; height: number } | null {
	const sw = source.width ?? 0;
	const sh = source.height ?? 0;
	const width = num(values.width);
	const height = num(values.height);
	if (width && height) return { width, height };
	if (!sw || !sh) return null;
	if (width) return { width, height: Math.max(1, Math.round((sh * width) / sw)) };
	if (height) return { width: Math.max(1, Math.round((sw * height) / sh)), height };
	return { width: sw, height: sh };
}

export function computeReadout(
	tool: Pick<ToolDTO, 'id' | 'cardinality' | 'produces'>,
	values: Record<string, unknown>,
	sources: readonly MediaToolSource[]
): string | null {
	const first = sources[0];
	if (!first) return null;
	const duration = first.durationSeconds ?? 0;

	if (tool.cardinality === 'many_to_one') {
		const total = sources.reduce((sum, source) => sum + (source.durationSeconds ?? 0), 0);
		const frames = values.overlap === 'crossfade' ? (num(values.overlap_frames) ?? 0) : 0;
		const fps = first.fps && first.fps > 0 ? first.fps : 24;
		const overlap = (frames / fps) * Math.max(0, sources.length - 1);
		const length = total > 0 ? ` · ${formatReadoutTime(Math.max(0, total - overlap))}` : '';
		return `1 video from ${sources.length} clips${length}`;
	}

	if (tool.cardinality === 'one_to_many') {
		const part = num(values.part_seconds);
		if (!part || part <= 0 || duration <= 0) return null;
		const parts = Math.ceil(duration / part);
		return `${parts} ${parts === 1 ? 'part' : 'parts'} of ${formatReadoutTime(part)}`;
	}

	if (sources.length > 1) {
		return `${sources.length} items, each saved as a new item`;
	}

	switch (tool.id) {
		case 'frame':
			return join(['1 image', size(first.width, first.height)]);
		case 'speed': {
			const speed = num(values.speed);
			return join([
				size(first.width, first.height),
				speed && speed > 0 && duration > 0 ? formatReadoutTime(duration / speed) : null
			]);
		}
		case 'rotate': {
			const degrees = num(values.degrees) ?? Number(values.degrees);
			const swap = degrees === 90 || degrees === 270;
			return join([
				swap ? size(first.height, first.width) : size(first.width, first.height),
				duration > 0 ? formatReadoutTime(duration) : null
			]);
		}
		case 'resize': {
			const next = resizedSize(first, values);
			const even = first.kind === 'video';
			return join([
				next ? size(even ? evenFloor(next.width) : next.width, even ? evenFloor(next.height) : next.height) : null,
				duration > 0 ? formatReadoutTime(duration) : null
			]);
		}
		case 'fit_canvas': {
			const width = num(values.width);
			const height = num(values.height);
			return width && height ? size(width, height) : null;
		}
		case 'color_key':
			return join([size(first.width, first.height), 'PNG with transparency']);
		case 'crop_subject':
			return 'Sized to the subject';
		case 'flip':
			return join([
				size(first.width, first.height),
				duration > 0 ? formatReadoutTime(duration) : null
			]);
		default:
			return null;
	}
}
