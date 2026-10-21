import { describe, it, expect } from 'vitest';
import { getApiErrorMessage } from './logger';

function axiosError(data: unknown, message = 'Request failed with status code 400') {
	return { isAxiosError: true, message, response: { data } };
}

describe('getApiErrorMessage', () => {
	it('prefers the FastAPI detail.message shape over a bare message field', () => {
		const err = axiosError({ detail: { message: 'Attribute key already exists' } });
		expect(getApiErrorMessage(err, 'fallback')).toBe('Attribute key already exists');
	});

	it('falls back to a bare data.message when there is no detail', () => {
		const err = axiosError({ message: 'plain message' });
		expect(getApiErrorMessage(err, 'fallback')).toBe('plain message');
	});

	it('falls back to the axios error message when the response has no usable shape', () => {
		const err = axiosError({});
		expect(getApiErrorMessage(err, 'fallback')).toBe('Request failed with status code 400');
	});

	it('falls back to the provided default for a non-axios, non-Error value', () => {
		expect(getApiErrorMessage('nope', 'fallback')).toBe('nope');
		expect(getApiErrorMessage({}, 'fallback')).toBe('fallback');
	});

	it('reads a FastAPI validation array as the first field message', () => {
		const err = axiosError({ detail: [{ loc: ['body', 'name'], msg: 'Value error, name is required' }] });
		expect(getApiErrorMessage(err, 'fallback')).toBe('Name is required');
	});

	it('reads a string detail', () => {
		expect(getApiErrorMessage(axiosError({ detail: 'Group is in use' }), 'fallback')).toBe('Group is in use');
	});

	it('uses a plain Error message and reads response data from non-axios objects', () => {
		expect(getApiErrorMessage(new Error('boom'), 'fallback')).toBe('boom');
		expect(getApiErrorMessage({ response: { data: { message: 'mocked' } } }, 'fallback')).toBe('mocked');
	});
});
