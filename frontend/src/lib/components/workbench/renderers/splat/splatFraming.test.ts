import { describe, it, expect } from 'vitest';
import { frameFromCenters, sampleStride, fitDistance, directionToward } from './splatFraming';

function grid(min: number, max: number, steps: number): number[] {
	const out: number[] = [];
	for (let x = 0; x <= steps; x++)
		for (let y = 0; y <= steps; y++)
			for (let z = 0; z <= steps; z++)
				out.push(
					min + ((max - min) * x) / steps,
					min + ((max - min) * y) / steps,
					min + ((max - min) * z) / steps
				);
	return out;
}

describe('frameFromCenters', () => {
	it('centres on the middle of the points and takes half the diagonal as radius', () => {
		const framing = frameFromCenters(grid(-1, 1, 10), 0)!;
		expect(framing.center[0]).toBeCloseTo(0, 5);
		expect(framing.center[1]).toBeCloseTo(0, 5);
		expect(framing.center[2]).toBeCloseTo(0, 5);
		expect(framing.radius).toBeCloseTo(Math.sqrt(3), 5);
	});

	it('follows an offset cloud', () => {
		const framing = frameFromCenters(grid(2, 6, 8), 0)!;
		expect(framing.center[0]).toBeCloseTo(4, 5);
		expect(framing.center[2]).toBeCloseTo(4, 5);
	});

	it('ignores a few far outliers instead of framing the empty space around them', () => {
		const points = grid(-1, 1, 20);
		points.push(5000, 5000, 5000, -5000, -5000, -5000);
		const trimmed = frameFromCenters(points)!;
		const untrimmed = frameFromCenters(points, 0)!;
		expect(trimmed.radius).toBeLessThan(5);
		expect(untrimmed.radius).toBeGreaterThan(1000);
	});

	it('skips non-finite coordinates', () => {
		const points = [...grid(-1, 1, 6), Number.NaN, 0, 0, Number.POSITIVE_INFINITY, 0, 0];
		const framing = frameFromCenters(points, 0)!;
		expect(Number.isFinite(framing.radius)).toBe(true);
		expect(framing.radius).toBeCloseTo(Math.sqrt(3), 5);
	});

	it('never returns a zero radius, even for a single point', () => {
		const framing = frameFromCenters([3, 4, 5])!;
		expect(framing.center).toEqual([3, 4, 5]);
		expect(framing.radius).toBeGreaterThan(0);
	});

	it('returns null when there is nothing to frame', () => {
		expect(frameFromCenters([])).toBeNull();
		expect(frameFromCenters([Number.NaN, Number.NaN, Number.NaN])).toBeNull();
	});
});

describe('sampleStride', () => {
	it('keeps every point under the limit and thins out above it', () => {
		expect(sampleStride(1000, 200000)).toBe(1);
		expect(sampleStride(200000, 200000)).toBe(1);
		expect(sampleStride(200001, 200000)).toBe(2);
		expect(sampleStride(1_000_000, 200000)).toBe(5);
	});

	it('never returns less than 1', () => {
		expect(sampleStride(0, 200000)).toBe(1);
		expect(sampleStride(10, 0)).toBe(10);
	});
});

describe('fitDistance', () => {
	it('puts a unit sphere in view, at the vertical limit for a wide viewport', () => {
		expect(fitDistance(1, 60, 2, 1)).toBeCloseTo(1 / Math.sin(Math.PI / 6), 5);
	});

	it('backs off further when the viewport is narrower than tall', () => {
		expect(fitDistance(1, 50, 0.5)).toBeGreaterThan(fitDistance(1, 50, 2));
	});

	it('scales with the radius and the margin', () => {
		expect(fitDistance(4, 50, 1.5, 1.2)).toBeCloseTo(fitDistance(1, 50, 1.5, 1.2) * 4, 5);
		expect(fitDistance(1, 50, 1.5, 2)).toBeCloseTo(fitDistance(1, 50, 1.5, 1) * 2, 5);
	});

	it('treats a collapsed viewport as square instead of dividing by zero', () => {
		expect(Number.isFinite(fitDistance(1, 50, 0))).toBe(true);
	});
});

describe('directionToward', () => {
	it('points from the content toward the source camera at the origin', () => {
		expect(directionToward([0, 0, -4])).toEqual([0, 0, 1]);
	});

	it('is a unit vector', () => {
		const [x, y, z] = directionToward([3, -4, -12]);
		expect(Math.hypot(x, y, z)).toBeCloseTo(1, 6);
	});

	it('falls back to +z when the content is centred on the origin', () => {
		expect(directionToward([0, 0, 0])).toEqual([0, 0, 1]);
	});
});
