// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { mount, unmount } from 'svelte';
import { settle } from './helpers/settle';

vi.mock('$lib/services/api', async () => {
	const actual = await vi.importActual<typeof import('$lib/services/api')>('$lib/services/api');
	return {
		...actual,
		api: { ...actual.api, testHttpRequest: vi.fn(), getToken: vi.fn(() => null) }
	};
});

const api = (await import('$lib/services/api')).api;
const { default: KeyValueField } = await import('../../src/lib/components/form-fields/KeyValueField.svelte');
const { default: HttpRequestTestPanel } = await import(
	'../../src/routes/automations/[id]/components/HttpRequestTestPanel.svelte'
);

const mounted: Array<() => void> = [];

function render(component: any, props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const instance = mount(component, { target, props });
	mounted.push(() => {
		unmount(instance);
		target.remove();
	});
	return target;
}

afterEach(() => {
	while (mounted.length) mounted.pop()!();
	vi.clearAllMocks();
});

function type(input: HTMLInputElement, text: string) {
	input.value = text;
	input.dispatchEvent(new Event('input', { bubbles: true }));
}

describe('KeyValueField', () => {
	it('adds a row and reports the new list', async () => {
		const onChange = vi.fn();
		const target = render(KeyValueField, {
			name: 'headers',
			config: { title: 'Headers', secret_toggle: true },
			value: [],
			onChange
		});

		const add = [...target.querySelectorAll('button')].find((b) => b.textContent?.includes('Add row'))!;
		add.click();

		expect(onChange).toHaveBeenCalledWith('headers', [{ name: '', value: '' }]);
	});

	it('edits name and value and keeps the other cell', async () => {
		const onChange = vi.fn();
		const target = render(KeyValueField, {
			name: 'headers',
			config: { title: 'Headers' },
			value: [{ name: 'Accept', value: 'json' }],
			onChange
		});

		type(target.querySelector('[aria-label="Headers value 1"]') as HTMLInputElement, 'xml');

		expect(onChange).toHaveBeenLastCalledWith('headers', [{ name: 'Accept', value: 'xml' }]);
	});

	it('toggles a row to secret, which hides the value', async () => {
		const onChange = vi.fn();
		const target = render(KeyValueField, {
			name: 'headers',
			config: { title: 'Headers', secret_toggle: true },
			value: [{ name: 'X-Token', value: 'abc' }],
			onChange
		});

		const toggle = target.querySelector('[aria-pressed]') as HTMLButtonElement;
		toggle.click();

		expect(onChange).toHaveBeenLastCalledWith('headers', [
			{ name: 'X-Token', value: 'abc', secret: true }
		]);
	});

	it('renders a secret row as a password input and offers no toggle without the flag', async () => {
		const withSecret = render(KeyValueField, {
			name: 'headers',
			config: { title: 'Headers', secret_toggle: true },
			value: [{ name: 'Authorization', value: '***', secret: true }],
			onChange: vi.fn()
		});
		const plain = render(KeyValueField, {
			name: 'query',
			config: { title: 'Query' },
			value: [{ name: 'q', value: 'x' }],
			onChange: vi.fn()
		});

		expect(withSecret.querySelector('[aria-label="Headers value 1"]')?.getAttribute('type')).toBe('password');
		expect(plain.querySelector('[aria-pressed]')).toBeNull();
	});

	it('removes a row', async () => {
		const onChange = vi.fn();
		const target = render(KeyValueField, {
			name: 'query',
			config: { title: 'Query' },
			value: [
				{ name: 'a', value: '1' },
				{ name: 'b', value: '2' }
			],
			onChange
		});

		(target.querySelector('[aria-label="Remove Query row 1"]') as HTMLButtonElement).click();

		expect(onChange).toHaveBeenCalledWith('query', [{ name: 'b', value: '2' }]);
	});
});

describe('HttpRequestTestPanel', () => {
	const config = { url: 'https://api.example.com', method: 'GET' };

	it('runs the request with the node config and shows status and body', async () => {
		vi.mocked(api.testHttpRequest).mockResolvedValue({
			success: true,
			data: {
				status: 200,
				ok: true,
				headers: {},
				body: { id: 7 },
				truncated: false,
				url: 'https://api.example.com',
				attempts: 1,
				pick: null
			}
		} as never);
		const target = render(HttpRequestTestPanel, { config, automationId: 'a1', nodeId: 'n1' });

		(target.querySelector('textarea') as HTMLTextAreaElement).value = '{"x": 1}';
		target.querySelector('textarea')!.dispatchEvent(new Event('input', { bubbles: true }));
		const run = [...target.querySelectorAll('button')].find((b) => b.textContent?.includes('Test request'))!;
		run.click();
		await settle();

		expect(api.testHttpRequest).toHaveBeenCalledWith({
			config,
			automation_id: 'a1',
			node_id: 'n1',
			event: { x: 1 }
		});
		expect(target.querySelector('[data-http-test-result]')?.textContent).toContain('200');
		expect(target.querySelector('[data-http-test-body]')?.textContent).toContain('"id": 7');
	});

	it('shows the server message when the request is refused', async () => {
		vi.mocked(api.testHttpRequest).mockResolvedValue({
			success: false,
			message: 'Requests to 127.0.0.1 are blocked.'
		} as never);
		const target = render(HttpRequestTestPanel, { config, nodeId: 'n1' });

		[...target.querySelectorAll('button')].find((b) => b.textContent?.includes('Test request'))!.click();
		await settle();

		expect(target.querySelector('[data-http-test-error]')?.textContent).toContain('blocked');
		expect(target.querySelector('[data-http-test-result]')).toBeNull();
	});

	it('rejects a sample event that is not a JSON object without calling the server', async () => {
		const target = render(HttpRequestTestPanel, { config, nodeId: 'n1' });

		const textarea = target.querySelector('textarea') as HTMLTextAreaElement;
		textarea.value = '[1]';
		textarea.dispatchEvent(new Event('input', { bubbles: true }));
		[...target.querySelectorAll('button')].find((b) => b.textContent?.includes('Test request'))!.click();
		await settle();

		expect(api.testHttpRequest).not.toHaveBeenCalled();
		expect(target.querySelector('[data-http-test-error]')?.textContent).toContain('JSON object');
	});
});
