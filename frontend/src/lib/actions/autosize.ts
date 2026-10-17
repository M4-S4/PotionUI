export function fitTextareaHeight(node: HTMLTextAreaElement): void {
	node.style.height = 'auto';
	const styles = getComputedStyle(node);
	const border = parseFloat(styles.borderTopWidth) + parseFloat(styles.borderBottomWidth);
	node.style.height = `${node.scrollHeight + (styles.boxSizing === 'border-box' ? border : 0)}px`;
}

export default function autosize(node: HTMLTextAreaElement, initial?: unknown) {
	let lastValue = initial;
	let lastWidth = node.clientWidth;

	function fit() {
		lastWidth = node.clientWidth;
		fitTextareaHeight(node);
	}

	fit();
	const observer =
		typeof ResizeObserver === 'function'
			? new ResizeObserver(() => {
					if (node.clientWidth !== lastWidth) fit();
				})
			: null;
	observer?.observe(node);
	return {
		update(value?: unknown) {
			if (value === lastValue && node.clientWidth === lastWidth) return;
			lastValue = value;
			fit();
		},
		destroy() {
			observer?.disconnect();
		}
	};
}
