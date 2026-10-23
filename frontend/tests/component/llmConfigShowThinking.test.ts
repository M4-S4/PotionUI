import { describe, it, expect, vi, afterEach } from 'vitest';
import type { LLMConfig } from '$lib/types/llm';
import { settle } from './helpers/settle';

const mockGetLLMConfigurations = vi.fn();
const mockGetPreChatActions = vi.fn();
const mockGetLLMAssignmentSummary = vi.fn();
const mockUpdateLLMConfiguration = vi.fn();

vi.mock('$lib/services/api/index', () => ({
	api: {
		getLLMConfigurations: (...args: unknown[]) => mockGetLLMConfigurations(...args),
		getPreChatActions: (...args: unknown[]) => mockGetPreChatActions(...args)
	}
}));

vi.mock('$lib/services/admin-api', () => ({
	getLLMAssignmentSummary: (...args: unknown[]) => mockGetLLMAssignmentSummary(...args),
	updateLLMConfiguration: (...args: unknown[]) => mockUpdateLLMConfiguration(...args),
	createLLMConfiguration: vi.fn(),
	deleteLLMConfiguration: vi.fn()
}));

const { default: LLMConfigTab } = await import('../../src/routes/admin/components/LLMConfigTab.svelte');
const { createClassComponent } = await import('svelte/legacy');

function config(overrides: Partial<LLMConfig> = {}): LLMConfig {
	return {
		id: 'cfg-1',
		name: 'Reasoner',
		type: 'ollama',
		enabled: true,
		base_url: 'http://localhost:11434',
		api_key_set: false,
		model: 'qwen3',
		system_message: 'Be terse.',
		temperature: 0.7,
		max_tokens: 1000,
		timeout: 30,
		supports_vision: false,
		memory_reflection: true,
		provider_options: {},
		...overrides
	};
}

function mount() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({ component: LLMConfigTab as never, target, props: {} });
	return {
		target,
		destroy: () => {
			component.$destroy();
			target.remove();
		}
	};
}

let mounted: ReturnType<typeof mount> | undefined;

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
	vi.clearAllMocks();
});

async function open(configs: LLMConfig[]) {
	mockGetLLMConfigurations.mockResolvedValue({ success: true, data: { configurations: configs } });
	mockGetPreChatActions.mockResolvedValue({ success: true, data: { actions: [] } });
	mockGetLLMAssignmentSummary.mockResolvedValue({ success: true, data: {} });
	mockUpdateLLMConfiguration.mockResolvedValue({ success: true, data: configs[0] });
	mounted = mount();
	await settle();
	(mounted.target.querySelector('.dt-scroll > [role="row"]') as HTMLElement).click();
	await settle();
	return mounted.target;
}

function thinkingSwitch(target: HTMLElement) {
	return target.querySelector('input[role="switch"][aria-label="Show thinking to users"]') as HTMLInputElement;
}

async function save(target: HTMLElement) {
	const button = Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.includes('Save'));
	(button as HTMLButtonElement).click();
	await settle();
}

describe('admin LLM configuration: Show thinking to users', () => {
	it('is off for a configuration that predates the setting', async () => {
		const target = await open([config()]);

		expect(thinkingSwitch(target).checked).toBe(false);
	});

	it('reflects a stored on value', async () => {
		const target = await open([config({ show_thinking: true })]);

		expect(thinkingSwitch(target).checked).toBe(true);
	});

	it('sends the switched value on save', async () => {
		const target = await open([config()]);

		thinkingSwitch(target).click();
		await settle();
		await save(target);

		expect(mockUpdateLLMConfiguration).toHaveBeenCalledTimes(1);
		expect(mockUpdateLLMConfiguration.mock.calls[0][1].show_thinking).toBe(true);
	});

	it('turns a stored on value back off', async () => {
		const target = await open([config({ show_thinking: true })]);

		thinkingSwitch(target).click();
		await settle();
		await save(target);

		expect(mockUpdateLLMConfiguration.mock.calls[0][1].show_thinking).toBe(false);
	});

	it('says the switch only displays what the model produces and points at the thinking levels', async () => {
		const target = await open([config()]);

		expect(target.textContent).toContain('It only displays what the model produces');
		expect(target.textContent).toContain('choose a thinking level (Enabled, Low, Medium or High)');
	});
});
