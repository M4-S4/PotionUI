// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createClassComponent } from 'svelte/legacy';
import { tick } from 'svelte';
import { get } from 'svelte/store';

vi.mock('$lib/components/UnifiedAIChat.svelte', async () => ({
	default: (await import('./stubs/BlankStub.svelte')).default
}));

import GlobalChatPanel from '$lib/components/GlobalChatPanel.svelte';
import { chatPanelStore } from '$lib/stores/chatPanel';
import { registerDismissLayer, unregisterDismissLayer, resetLayerStackForTests } from '$lib/actions/layerStack';

let component: { $destroy: () => void } | undefined;

function pressEscape() {
	window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
}

beforeEach(() => {
	resetLayerStackForTests();
	chatPanelStore.reset();
	const target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({ component: GlobalChatPanel as never, target });
	chatPanelStore.open();
});

afterEach(() => {
	component?.$destroy();
	component = undefined;
	chatPanelStore.reset();
	document.body.innerHTML = '';
});

describe('GlobalChatPanel Escape', () => {
	it('closes the panel on Escape when no layer is open', async () => {
		await tick();
		pressEscape();
		expect(get(chatPanelStore).isOpen).toBe(false);
	});

	it('stays open while a dismiss layer is registered and closes once it is gone', async () => {
		await tick();
		const layer = registerDismissLayer({ node: document.createElement('div') });

		pressEscape();
		expect(get(chatPanelStore).isOpen).toBe(true);

		unregisterDismissLayer(layer);
		pressEscape();
		expect(get(chatPanelStore).isOpen).toBe(false);
	});
});
