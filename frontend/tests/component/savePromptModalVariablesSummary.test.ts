// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { flushSync } from 'svelte';
import type { Segment } from '$lib/types/segments';
import type { VariablesMap } from '$lib/utils/variableDefs';

const createPrompt = vi.fn().mockResolvedValue({ success: true, data: { id: 'p1' } });

vi.mock('$lib/services/api', () => ({
	api: { createPrompt: (...a: unknown[]) => createPrompt(...a) }
}));

const { default: SavePromptModal } = await import('../../src/lib/components/modals/SavePromptModal.svelte');
const { createClassComponent } = await import('svelte/legacy');

function mountModal(segments: Segment[], variables: VariablesMap) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: SavePromptModal as never,
		target,
		props: { isOpen: true, segments, variables }
	});
	flushSync();
	return { target, component, destroy: () => component.$destroy() };
}

function clickSave() {
	const save = [...document.body.querySelectorAll('button')].find((b) => b.textContent?.includes('Save Prompt'));
	save?.click();
}

let mounted: ReturnType<typeof mountModal> | undefined;

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
	createPrompt.mockClear();
	document.body.innerHTML = '';
});

const variables: VariablesMap = {
	music: { type: 'choice', options: ['hip hop', 'classical'], mode: 'shuffle', pinnedIndex: null },
	dance: {
		type: 'choice',
		mode: 'shuffle',
		pinnedIndex: null,
		options: [{ text: 'breaking', when: { var: 'music', values: ['hip hop'] } }, 'freestyle']
	},
	unused: { type: 'text', value: 'never referenced' }
};

describe('SavePromptModal variables', () => {
	it('summarises every defined variable, referenced or not', () => {
		const segments: Segment[] = [{ id: 's1', content: 'a ${dance} scene', enabled: true }];
		mounted = mountModal(segments, variables);

		expect(document.body.textContent).toContain('Includes 3 variables · 1 condition');
	});

	it('saves every defined variable including ones the text never references', async () => {
		const segments: Segment[] = [{ id: 's1', content: 'a plain scene', enabled: true }];
		mounted = mountModal(segments, variables);

		clickSave();
		await vi.waitFor(() => expect(createPrompt).toHaveBeenCalledTimes(1));

		expect(Object.keys(createPrompt.mock.calls[0][0].variables).sort()).toEqual(['dance', 'music', 'unused']);
	});

	it('saves null variables and shows no summary when none are defined', async () => {
		const segments: Segment[] = [{ id: 's1', content: 'a plain scene', enabled: true }];
		mounted = mountModal(segments, {});

		expect(document.body.textContent).not.toContain('Includes');
		clickSave();
		await vi.waitFor(() => expect(createPrompt).toHaveBeenCalledTimes(1));

		expect(createPrompt.mock.calls[0][0].variables).toBeNull();
	});
});
