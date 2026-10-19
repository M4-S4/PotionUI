// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';

const { default: AccountRegistrationForm } = await import(
	'../../src/lib/components/auth/AccountRegistrationForm.svelte'
);

let component: ReturnType<typeof mount> | null = null;

afterEach(() => {
	if (component) unmount(component);
	component = null;
	document.body.innerHTML = '';
});

function fill(id: string, value: string) {
	const input = document.getElementById(id) as HTMLInputElement;
	input.value = value;
	input.dispatchEvent(new Event('input', { bubbles: true }));
}

async function submitWith(result: { success: boolean; error?: string; fieldErrors?: Record<string, string> }) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	component = mount(AccountRegistrationForm as any, {
		target,
		props: {
			mode: 'register',
			submitLabel: 'Create',
			submitLoadingLabel: 'Creating',
			onSubmit: async () => result
		}
	});
	flushSync();
	fill('username', 'owner');
	fill('email', 'admin@potionui.local');
	fill('password', 'longenough1');
	fill('confirmPassword', 'longenough1');
	flushSync();
	target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
	await new Promise((resolve) => setTimeout(resolve, 0));
	flushSync();
	return target;
}

describe('AccountRegistrationForm server errors', () => {
	it('shows a server field error under the email field', async () => {
		const target = await submitWith({
			success: false,
			error: 'Email rejected',
			fieldErrors: { email: 'Value is not a valid email address: nope' }
		});
		const message = target.querySelector('#email-error');
		expect(message?.textContent).toContain('not a valid email address');
		expect(document.getElementById('email')?.getAttribute('aria-invalid')).toBe('true');
		expect(target.querySelector('[role="alert"]:not(#email-error)')).toBeNull();
	});

	it('shows a form-level error for failures without a field', async () => {
		const target = await submitWith({ success: false, error: 'Registration is closed' });
		expect(target.textContent).toContain('Registration is closed');
		expect(target.querySelector('#email-error')).toBeNull();
	});

	it('shows the form-level error when the field error has no input', async () => {
		const target = await submitWith({
			success: false,
			error: 'Invalid setup token',
			fieldErrors: { claim_token: 'Invalid setup token' }
		});
		expect(target.textContent).toContain('Invalid setup token');
	});
});
