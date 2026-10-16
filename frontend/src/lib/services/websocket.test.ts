// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';

const warn = vi.hoisted(() => vi.fn());
vi.mock('$lib/utils/logger', () => ({ logger: { warn, error: vi.fn(), info: vi.fn(), debug: vi.fn() } }));
vi.mock('$lib/stores/auth', () => ({ authStore: { expireSession: vi.fn() } }));
const tokenRef = vi.hoisted(() => ({ current: null as string | null }));
vi.mock('$lib/services/api/index', () => ({ api: { getToken: () => tokenRef.current } }));

import { authStore } from '$lib/stores/auth';
import { createGenerationSocket } from './websocket';

describe('createGenerationSocket', () => {
	it('expires the session on an auth-failed close (4001), same as every other generation socket', () => {
		const ws = createGenerationSocket();

		// Simulate the server closing the connection with the auth-failure code,
		// which WebSocketService.onClose() turns into the onAuthFailed callback.
		(ws as unknown as { onClose(event: CloseEvent): void }).onClose({ code: 4001 } as CloseEvent);

		expect(authStore.expireSession).toHaveBeenCalledTimes(1);
	});
});

describe('generation socket reconnect', () => {
	class RecordingSocket {
		static readonly CONNECTING = 0;
		static readonly OPEN = 1;
		static readonly CLOSING = 2;
		static readonly CLOSED = 3;
		static urls: string[] = [];
		readyState = RecordingSocket.CONNECTING;
		onopen: (() => void) | null = null;
		onmessage: ((event: MessageEvent) => void) | null = null;
		onerror: ((event: Event) => void) | null = null;
		onclose: ((event: CloseEvent) => void) | null = null;
		constructor(url: string) {
			RecordingSocket.urls.push(url);
		}
		close(): void {
			this.readyState = RecordingSocket.CLOSED;
		}
		send(): void {}
	}

	afterEach(() => {
		vi.useRealTimers();
		vi.unstubAllGlobals();
		tokenRef.current = null;
		RecordingSocket.urls = [];
	});

	it('reconnects with the token that is current at that moment, not the one it was created with', async () => {
		vi.useFakeTimers();
		vi.stubGlobal('WebSocket', RecordingSocket);
		tokenRef.current = 'first-token';
		const ws = createGenerationSocket();

		ws.connect();
		expect(RecordingSocket.urls[0]).toContain('token=first-token');

		tokenRef.current = 'renewed-token';
		const socket = (ws as unknown as { ws: RecordingSocket }).ws;
		socket.readyState = RecordingSocket.CLOSED;
		socket.onclose?.({ code: 1006 } as CloseEvent);
		await vi.advanceTimersByTimeAsync(5000);

		expect(RecordingSocket.urls).toHaveLength(2);
		expect(RecordingSocket.urls[1]).toContain('token=renewed-token');
		expect(RecordingSocket.urls[1]).not.toContain('first-token');
		ws.disconnect();
	});
});


describe('generation socket message routing', () => {
	function deliver(message: Record<string, unknown>): void {
		const ws = createGenerationSocket();
		(ws as unknown as { onMessage(m: unknown): void }).onMessage(message);
	}

	beforeEach(() => warn.mockClear());

	it('ignores the id-less status snapshot without warning', () => {
		deliver({ type: 'status_update', data: { id: 'g1', status: 'running' } });
		expect(warn).not.toHaveBeenCalled();
	});

	it('still warns about other messages that carry no generation id', () => {
		deliver({ type: 'gallery_update', data: {} });
		expect(warn).toHaveBeenCalledWith('Message gallery_update has no generation_id');
	});
});
