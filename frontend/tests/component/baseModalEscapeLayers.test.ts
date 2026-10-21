// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createClassComponent } from 'svelte/legacy';
import BaseModal from '$lib/components/modals/BaseModal.svelte';
import dismiss from '$lib/actions/dismiss';
import { resetLayerStackForTests } from '$lib/actions/layerStack';

let component: ReturnType<typeof createClassComponent> | undefined;

function openModal(props: Record<string, unknown> = {}) {
	const onClose = vi.fn();
	const target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({
		component: BaseModal as never,
		target,
		props: { isOpen: true, title: 'Details', ...props }
	});
	component.$on('close', onClose);
	return onClose;
}

function pressEscape() {
	window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
}

beforeEach(() => {
	resetLayerStackForTests();
});

afterEach(() => {
	component?.$destroy();
	component = undefined;
	document.body.innerHTML = '';
	document.body.style.overflow = '';
});

describe('BaseModal as a dismiss layer', () => {
	it('closes on Escape', async () => {
		const onClose = openModal();
		await Promise.resolve();
		pressEscape();
		expect(onClose).toHaveBeenCalledTimes(1);
	});

	it('ignores Escape when the caller owns it', async () => {
		const onClose = openModal({ handleEscapeKey: false });
		await Promise.resolve();
		pressEscape();
		expect(onClose).not.toHaveBeenCalled();
	});

	it('lets a popover opened above the modal take Escape first', async () => {
		const onClose = openModal();
		await Promise.resolve();
		const popover = document.createElement('div');
		document.body.appendChild(popover);
		const onDismiss = vi.fn();
		const layer = dismiss(popover, { onDismiss });

		pressEscape();
		expect(onDismiss).toHaveBeenCalledWith('escape', expect.any(Event));
		expect(onClose).not.toHaveBeenCalled();

		layer.destroy();
		pressEscape();
		expect(onClose).toHaveBeenCalledTimes(1);
	});

	it('does not close on an outside pointerdown', async () => {
		const onClose = openModal();
		await Promise.resolve();
		document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }));
		expect(onClose).not.toHaveBeenCalled();
	});
});
