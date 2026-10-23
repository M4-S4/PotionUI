import { describe, expect, it } from 'vitest';
import { parseApiError } from './apiErrors';

const validation = (items: unknown[]) => ({
	response: {
		status: 422,
		data: { success: false, error: 'validation_error', message: 'Request validation failed', detail: items }
	},
	message: 'Request failed with status code 422'
});

describe('parseApiError', () => {
	it('maps a 422 detail list to per-field messages', () => {
		const parsed = parseApiError(
			validation([
				{
					type: 'value_error',
					loc: ['body', 'email'],
					msg: 'value is not a valid email address: An email address must have an @-sign.'
				}
			])
		);
		expect(parsed.fieldErrors).toEqual({
			email: 'Value is not a valid email address: An email address must have an @-sign.'
		});
		expect(parsed.message).toBe(parsed.fieldErrors.email);
	});

	it('strips the pydantic value-error prefix', () => {
		const parsed = parseApiError(
			validation([{ loc: ['body', 'password'], msg: 'Value error, Password is too short' }])
		);
		expect(parsed.fieldErrors.password).toBe('Password is too short');
	});

	it('falls back to the response message when no field is named', () => {
		const parsed = parseApiError(validation([{ loc: ['body'], msg: 'Field required' }]));
		expect(parsed.fieldErrors).toEqual({});
		expect(parsed.message).toBe('Request validation failed');
	});

	it('unwraps string and wrapped details', () => {
		expect(parseApiError({ response: { data: { detail: 'Nope' } } }).message).toBe('Nope');
		expect(parseApiError({ response: { data: { detail: { message: 'Closed' } } } }).message).toBe('Closed');
	});

	it('falls back to the transport message', () => {
		expect(parseApiError(new Error('Network Error')).message).toBe('Network Error');
	});
});
