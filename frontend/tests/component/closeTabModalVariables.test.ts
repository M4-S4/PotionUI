// @vitest-environment jsdom
import { describe, it, expect, afterEach, beforeEach } from 'vitest';
import { tick } from 'svelte';

const { default: CloseTabModal } = await import('../../src/lib/tabs/CloseTabModal.svelte');
const { createClassComponent } = await import('svelte/legacy');
const { tabsStore } = await import('../../src/lib/stores/tabs');
const { requestCloseTab, cancelCloseTab } = await import('../../src/lib/tabs/closeConfirm');

let component: { $destroy(): void } | undefined;

beforeEach(() => {
	tabsStore.reset();
	tabsStore.addTab();
});

afterEach(() => {
	cancelCloseTab();
	component?.$destroy();
	component = undefined;
	document.body.innerHTML = '';
});

function mount() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({ component: CloseTabModal as never, target });
}

function firstTabId(): string {
	let id = '';
	tabsStore.subscribe((state) => (id = state.tabs[0].id))();
	return id;
}

describe('CloseTabModal variables warning', () => {
	it('names the unsaved variables when the tab was never saved', async () => {
		const id = firstTabId();
		tabsStore.updateTab(id, { variables: { mood: { type: 'text', value: 'noir' } } });
		mount();
		requestCloseTab(id);
		await tick();
		expect(document.body.textContent).toContain("This tab's variables aren't saved anywhere else.");
	});

	it('does not mention variables for a tab bound to a session', async () => {
		const id = firstTabId();
		tabsStore.updateTab(id, {
			variables: { mood: { type: 'text', value: 'noir' } },
			selectedSessionId: 'session-1'
		});
		mount();
		requestCloseTab(id);
		await tick();
		expect(document.body.textContent).not.toContain("aren't saved anywhere else");
	});

	it('does not mention variables for a tab without any', async () => {
		const id = firstTabId();
		mount();
		requestCloseTab(id);
		await tick();
		expect(document.body.textContent).toContain('will be closed');
		expect(document.body.textContent).not.toContain("aren't saved anywhere else");
	});
});
