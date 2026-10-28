import { describe, it, expect } from 'vitest';
import { thresholdMaskPixels } from './maskThreshold';

describe('thresholdMaskPixels', () => {
	it('drops the faint residue an eraser or an antialiased rim leaves and keeps painted pixels white', () => {
		const pixels = new Uint8ClampedArray([255, 255, 255, 255, 3, 3, 3, 255, 64, 64, 64, 255, 200, 200, 200, 255, 0, 0, 0, 255]);
		const lit = thresholdMaskPixels(pixels);
		expect(lit).toBe(2);
		expect(Array.from(pixels)).toEqual([255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255]);
	});
});
