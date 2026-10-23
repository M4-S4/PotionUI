import { describe, it, expect, vi } from 'vitest';
import { adminWebSocket } from './adminWebsocket';

function receive(message: Record<string, unknown>) {
	(adminWebSocket as unknown as { onMessage: (m: unknown) => void }).onMessage(message);
}

describe('indexing status frames', () => {
	it('hands a pushed indexing_status to every listener until it unsubscribes', () => {
		const first = vi.fn();
		const second = vi.fn();
		const offFirst = adminWebSocket.onIndexingStatus(first);
		const offSecond = adminWebSocket.onIndexingStatus(second);

		receive({ type: 'indexing_status', status: { state: 'indexing', processed: 3 } });
		expect(first).toHaveBeenCalledWith({ state: 'indexing', processed: 3 });
		expect(second).toHaveBeenCalledTimes(1);

		offFirst();
		receive({ type: 'indexing_status', status: { state: 'done' } });
		expect(first).toHaveBeenCalledTimes(1);
		expect(second).toHaveBeenCalledTimes(2);
		offSecond();
	});

	it('tells connection listeners when the server confirms the connection', () => {
		const connected = vi.fn();
		const off = adminWebSocket.onConnected(connected);

		receive({ type: 'connection_established' });

		expect(connected).toHaveBeenCalledTimes(1);
		off();
	});
});
