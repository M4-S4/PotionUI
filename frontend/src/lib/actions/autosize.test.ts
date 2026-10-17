// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import autosize, { fitTextareaHeight } from './autosize';

function textarea(scrollHeight: number): HTMLTextAreaElement {
	const node = document.createElement('textarea');
	document.body.appendChild(node);
	Object.defineProperty(node, 'scrollHeight', { configurable: true, get: () => scrollHeight });
	return node;
}

describe('autosize', () => {
	it('sizes the textarea to its content', () => {
		const node = textarea(63);
		fitTextareaHeight(node);
		expect(node.style.height).toBe('63px');
	});

	it('follows the content when it changes', () => {
		let height = 48;
		const node = document.createElement('textarea');
		document.body.appendChild(node);
		Object.defineProperty(node, 'scrollHeight', { configurable: true, get: () => height });
		const action = autosize(node, 'a');
		expect(node.style.height).toBe('48px');
		height = 68;
		action.update('ab');
		expect(node.style.height).toBe('68px');
		action.destroy();
	});

	it('does not touch the height when neither the text nor the width changed', () => {
		let height = 48;
		const node = document.createElement('textarea');
		document.body.appendChild(node);
		Object.defineProperty(node, 'scrollHeight', { configurable: true, get: () => height });
		const action = autosize(node, 'same');
		height = 90;
		action.update('same');
		expect(node.style.height).toBe('48px');
		action.destroy();
	});

	it('does not change height on focus or blur', () => {
		const node = textarea(48);
		const action = autosize(node, 'a');
		const before = node.style.height;
		node.dispatchEvent(new Event('focus'));
		node.dispatchEvent(new Event('blur'));
		expect(node.style.height).toBe(before);
		action.destroy();
	});
});
