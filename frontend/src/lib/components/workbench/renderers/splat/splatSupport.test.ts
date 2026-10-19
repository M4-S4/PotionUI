import { describe, it, expect } from 'vitest';
import { SplatLoadError, splatErrorFromStatus, splatErrorMessage } from './splatSupport';

describe('splatErrorFromStatus', () => {
	it('tells a missing file from a refused one from a server failure', () => {
		expect(splatErrorFromStatus(404).kind).toBe('not-found');
		expect(splatErrorFromStatus(410).kind).toBe('not-found');
		expect(splatErrorFromStatus(401).kind).toBe('forbidden');
		expect(splatErrorFromStatus(403).kind).toBe('forbidden');
		const failure = splatErrorFromStatus(502);
		expect(failure.kind).toBe('http');
		expect(failure.status).toBe(502);
	});
});

describe('splatErrorMessage', () => {
	it('gives each failure its own readable sentence', () => {
		const kinds = ['not-found', 'forbidden', 'network', 'decode', 'empty', 'timeout', 'webgl'] as const;
		const messages = kinds.map((kind) => splatErrorMessage(new SplatLoadError(kind)));
		expect(new Set(messages).size).toBe(kinds.length);
		for (const message of messages) expect(message).toMatch(/^[A-Z].*\.$/);
	});

	it('names the HTTP status for a server failure', () => {
		expect(splatErrorMessage(splatErrorFromStatus(500))).toContain('500');
	});

	it('describes any other error as an unreadable file rather than leaking its text', () => {
		const message = splatErrorMessage(new TypeError('Cannot read properties of undefined'));
		expect(message).toContain('could not be displayed');
		expect(message).not.toContain('undefined');
	});
});
