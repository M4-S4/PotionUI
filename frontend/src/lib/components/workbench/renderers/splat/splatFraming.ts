import type { SourceView } from './splatHeader';

export type Vec3 = [number, number, number];

export interface SplatFraming {
	center: Vec3;
	radius: number;
}

const MIN_RADIUS = 1e-3;

function quantile(sorted: Float32Array, q: number): number {
	const position = q * (sorted.length - 1);
	const lower = Math.floor(position);
	const upper = Math.min(lower + 1, sorted.length - 1);
	return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

export function frameFromCenters(centers: ArrayLike<number>, trim = 0.01): SplatFraming | null {
	const count = Math.floor(centers.length / 3);
	if (count === 0) return null;

	const lows: Vec3 = [0, 0, 0];
	const highs: Vec3 = [0, 0, 0];
	for (let axis = 0; axis < 3; axis++) {
		const values = new Float32Array(count);
		let kept = 0;
		for (let i = 0; i < count; i++) {
			const value = centers[i * 3 + axis];
			if (Number.isFinite(value)) values[kept++] = value;
		}
		if (kept === 0) return null;
		const sorted = values.subarray(0, kept).sort();
		lows[axis] = quantile(sorted, trim);
		highs[axis] = quantile(sorted, 1 - trim);
	}

	const center: Vec3 = [
		(lows[0] + highs[0]) / 2,
		(lows[1] + highs[1]) / 2,
		(lows[2] + highs[2]) / 2
	];
	const diagonal = Math.hypot(highs[0] - lows[0], highs[1] - lows[1], highs[2] - lows[2]);
	return { center, radius: Math.max(diagonal / 2, MIN_RADIUS) };
}

export function sampleStride(total: number, limit: number): number {
	return Math.max(1, Math.ceil(total / Math.max(1, limit)));
}

export function fitDistance(radius: number, fovYDegrees: number, aspect: number, margin = 1.2): number {
	const halfVertical = (fovYDegrees * Math.PI) / 360;
	const halfHorizontal = Math.atan(Math.tan(halfVertical) * (aspect > 0 ? aspect : 1));
	return (radius / Math.sin(Math.min(halfVertical, halfHorizontal))) * margin;
}

export function directionToward(from: Vec3, to: Vec3 = [0, 0, 0]): Vec3 {
	const dx = to[0] - from[0];
	const dy = to[1] - from[1];
	const dz = to[2] - from[2];
	const length = Math.hypot(dx, dy, dz);
	if (length < 1e-6) return [0, 0, 1];
	return [dx / length, dy / length, dz / length];
}

export const DEFAULT_FOV_Y = 50;

export interface CameraView {
	position: Vec3;
	target: Vec3;
	fov: number;
	near: number;
	far: number;
	minDistance: number;
	maxDistance: number;
}

export function viewFromFraming(
	center: Vec3,
	radius: number,
	aspect: number,
	fov = DEFAULT_FOV_Y
): CameraView {
	const [dx, dy, dz] = directionToward(center);
	const distance = fitDistance(radius, fov, aspect);
	return {
		position: [center[0] + dx * distance, center[1] + dy * distance, center[2] + dz * distance],
		target: center,
		fov,
		near: radius * 0.01,
		far: distance + radius * 50,
		minDistance: radius * 0.05,
		maxDistance: distance * 8
	};
}

export function viewFromSource(source: SourceView): CameraView {
	return {
		position: [0, 0, 0],
		target: [0, 0, -source.depth],
		fov: source.fovY,
		near: source.depth * 0.01,
		far: source.depth * 100,
		minDistance: source.depth * 0.05,
		maxDistance: source.depth * 8
	};
}
