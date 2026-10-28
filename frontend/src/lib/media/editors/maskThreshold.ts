export function thresholdMaskPixels(pixels: Uint8ClampedArray, cutoff = 128): number {
	let lit = 0;
	for (let i = 0; i < pixels.length; i += 4) {
		const on = pixels[i] >= cutoff;
		const value = on ? 255 : 0;
		pixels[i] = value;
		pixels[i + 1] = value;
		pixels[i + 2] = value;
		pixels[i + 3] = 255;
		if (on) lit++;
	}
	return lit;
}
