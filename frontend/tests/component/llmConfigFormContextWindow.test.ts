import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount as mountComponent, unmount } from 'svelte';
import type { LLMConfigFormData } from '../../src/routes/admin/components/LLMConfigForm.svelte';
import { reactiveProps } from './stubs/reactiveProps.svelte';

vi.mock('$lib/services/api/index', () => ({
	api: {
		listNativeCheckpoints: vi.fn(async () => ({ success: true, data: [] })),
		getLLMProviderTypes: vi.fn(async () => ({ success: true, data: { types: [] } }))
	}
}));

const { default: LLMConfigForm } = await import(
	'../../src/routes/admin/components/LLMConfigForm.svelte'
);

const OLLAMA_NOTE = 'Context Size is blank';

function makeDraft(overrides: Partial<LLMConfigFormData> = {}): LLMConfigFormData {
	return {
		name: 'Test config',
		type: 'openai',
		model: 'gpt-4o',
		api_key: '',
		base_url: '',
		enabled: true,
		supports_vision: false,
		disable_system_prompt: false,
		memory_reflection: true,
		system_message: 'You are helpful.',
		temperature: 0.7,
		max_tokens: 1000,
		timeout: 30,
		provider_options: {},
		...overrides
	};
}

function mount(initial: LLMConfigFormData, layout: 'panel' | 'plain' = 'plain') {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const live = reactiveProps<Record<string, unknown>>({ draft: initial, mode: 'edit', layout, idPrefix: 'test' });
	const component = mountComponent(LLMConfigForm as any, { target, props: live.props as any });
	flushSync();
	return {
		target,
		get draft() {
			return live.props.draft as LLMConfigFormData;
		},
		destroy: () => {
			unmount(component);
			target.remove();
		}
	};
}

async function settle() {
	for (let i = 0; i < 8; i++) await new Promise((resolve) => setTimeout(resolve, 0));
	flushSync();
}

async function openSection(target: HTMLElement, title: string) {
	const heading = Array.from(target.querySelectorAll('h3')).find((h) => h.textContent?.trim() === title);
	expect(heading, `${title} heading`).toBeTruthy();
	const section = heading!.closest('section') ?? heading!.parentElement!.parentElement!;
	const toggle = section.querySelector('button[aria-expanded="false"]') as HTMLButtonElement | null;
	if (toggle) {
		toggle.click();
		await settle();
	}
	return section as HTMLElement;
}

function numberInput(root: HTMLElement, id: string) {
	return root.querySelector(`#${id}`) as HTMLInputElement;
}

function setValue(input: HTMLInputElement, value: string) {
	input.value = value;
	input.dispatchEvent(new Event('input', { bubbles: true }));
}

let mounted: ReturnType<typeof mount> | undefined;

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
	vi.clearAllMocks();
});

describe('LLMConfigForm — OpenAI context window', () => {
	it('renders a blank Context window (tokens) field inside OpenAI Options', async () => {
		mounted = mount(makeDraft());
		await settle();

		const section = await openSection(mounted.target, 'OpenAI Options');
		const label = Array.from(section.querySelectorAll('label')).find((l) => l.textContent?.trim() === 'Context window (tokens)');
		expect(label).toBeTruthy();
		expect(label?.getAttribute('for')).toBe('test-openai-context_window');

		const input = numberInput(section, 'test-openai-context_window');
		expect(input).toBeTruthy();
		expect(input.type).toBe('number');
		expect(input.value).toBe('');
		expect(input.placeholder).toBe('auto');
		expect('context_window' in mounted.draft.provider_options).toBe(false);
		expect(section.textContent).toContain('llama-server -c');
		expect(mounted.target.textContent).not.toContain(OLLAMA_NOTE);
	});

	it('renders the field in the panel layout as well', async () => {
		mounted = mount(makeDraft({ provider_options: { context_window: 32768 } }), 'panel');
		await settle();

		const section = await openSection(mounted.target, 'OpenAI Options');
		expect(numberInput(section, 'test-openai-context_window').value).toBe('32768');
	});

	it('stores 16384 as a number in provider_options and removes the key when cleared', async () => {
		mounted = mount(makeDraft());
		await settle();

		const section = await openSection(mounted.target, 'OpenAI Options');
		const input = numberInput(section, 'test-openai-context_window');

		setValue(input, '16384');
		await settle();
		expect(mounted.draft.provider_options.context_window).toBe(16384);

		setValue(input, '');
		await settle();
		expect('context_window' in mounted.draft.provider_options).toBe(false);
	});

	it('shows a saved value and leaves the other OpenAI options untouched when editing it', async () => {
		mounted = mount(makeDraft({ provider_options: { context_window: 8192, seed: 7 } }));
		await settle();

		const section = await openSection(mounted.target, 'OpenAI Options');
		const input = numberInput(section, 'test-openai-context_window');
		expect(input.value).toBe('8192');

		setValue(input, '16384');
		await settle();
		expect(mounted.draft.provider_options.context_window).toBe(16384);
		expect(mounted.draft.provider_options.seed).toBe(7);
	});
});

describe('LLMConfigForm — Ollama blank Context Size note', () => {
	it('shows the note while num_ctx is unset and hides it once a value is entered', async () => {
		mounted = mount(makeDraft({ type: 'ollama', model: 'llama3' }));
		await settle();

		const section = await openSection(mounted.target, 'Ollama Options');
		expect(section.textContent).toContain(OLLAMA_NOTE);
		expect(section.textContent).toContain('4,096 tokens');

		const input = numberInput(section, 'test-ollama-num_ctx');
		setValue(input, '16384');
		await settle();
		expect(mounted.draft.provider_options.num_ctx).toBe(16384);
		expect(section.textContent).not.toContain(OLLAMA_NOTE);

		setValue(input, '');
		await settle();
		expect('num_ctx' in mounted.draft.provider_options).toBe(false);
		expect(section.textContent).toContain(OLLAMA_NOTE);
	});

	it('does not show the note when the saved config already has num_ctx', async () => {
		mounted = mount(makeDraft({ type: 'ollama', model: 'llama3', provider_options: { num_ctx: 8192 } }));
		await settle();

		const section = await openSection(mounted.target, 'Ollama Options');
		expect(numberInput(section, 'test-ollama-num_ctx').value).toBe('8192');
		expect(section.textContent).not.toContain(OLLAMA_NOTE);
	});
});

describe('LLMConfigForm — native context window unchanged', () => {
	it('still renders the native field and round-trips it without any OpenAI field', async () => {
		mounted = mount(makeDraft({ type: 'native', model: 'qwen3-tiny' }));
		await settle();

		const section = await openSection(mounted.target, 'Native Options');
		const input = numberInput(section, 'test-native-context_window');
		expect(input).toBeTruthy();
		expect(input.value).toBe('');
		expect(mounted.target.querySelector('#test-openai-context_window')).toBeNull();

		setValue(input, '4096');
		await settle();
		expect(mounted.draft.provider_options.context_window).toBe(4096);

		setValue(input, '');
		await settle();
		expect('context_window' in mounted.draft.provider_options).toBe(false);
	});
});
