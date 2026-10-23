export function stubClientWidth(width: number): () => void {
	const original = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth');
	Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
		configurable: true,
		get: () => width
	});
	return () => {
		if (original) Object.defineProperty(HTMLElement.prototype, 'clientWidth', original);
		else delete (HTMLElement.prototype as { clientWidth?: number }).clientWidth;
	};
}
