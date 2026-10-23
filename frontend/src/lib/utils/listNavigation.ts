export function stepIndex(current: number, delta: number, length: number): number {
	if (length <= 0) return 0;
	return (((current + delta) % length) + length) % length;
}
