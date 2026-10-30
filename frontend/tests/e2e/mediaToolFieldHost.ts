import type { Locator, Page } from '@playwright/test';
import { loginAsOwner } from './helpers';
import { mockMediaTools, solidPng, type MediaToolLog, type MediaToolMockOptions } from './mediaToolMocks';

export type PreparedLog = MediaToolLog & { hasVideo: boolean };

function fixtureWav(seconds: number): Buffer {
	const rate = 8000;
	const samples = rate * seconds;
	const data = Buffer.alloc(samples * 2);
	for (let i = 0; i < samples; i++) data.writeInt16LE(Math.round(Math.sin(i / 12) * 6000), i * 2);
	const header = Buffer.alloc(44);
	header.write('RIFF', 0);
	header.writeUInt32LE(36 + data.length, 4);
	header.write('WAVEfmt ', 8);
	header.writeUInt32LE(16, 16);
	header.writeUInt16LE(1, 20);
	header.writeUInt16LE(1, 22);
	header.writeUInt32LE(rate, 24);
	header.writeUInt32LE(rate * 2, 28);
	header.writeUInt16LE(2, 32);
	header.writeUInt16LE(16, 34);
	header.write('data', 36);
	header.writeUInt32LE(data.length, 40);
	return Buffer.concat([header, data]);
}

async function recordWebm(page: Page): Promise<Buffer | null> {
	const encoded = await page.evaluate(async () => {
		if (typeof MediaRecorder === 'undefined') return null;
		const canvas = document.createElement('canvas');
		canvas.width = 160;
		canvas.height = 90;
		const context = canvas.getContext('2d');
		if (!context) return null;
		const recorder = new MediaRecorder(canvas.captureStream(10), { mimeType: 'video/webm;codecs=vp8' });
		const chunks: Blob[] = [];
		recorder.ondataavailable = (event) => chunks.push(event.data);
		const stopped = new Promise<void>((resolve) => (recorder.onstop = () => resolve()));
		recorder.start();
		for (let i = 0; i < 8; i++) {
			context.fillStyle = `hsl(${i * 40} 50% 40%)`;
			context.fillRect(0, 0, 160, 90);
			await new Promise((resolve) => setTimeout(resolve, 100));
		}
		recorder.stop();
		await stopped;
		const bytes = new Uint8Array(await new Blob(chunks).arrayBuffer());
		let binary = '';
		for (const byte of bytes) binary += String.fromCharCode(byte);
		return btoa(binary);
	});
	return encoded ? Buffer.from(encoded, 'base64') : null;
}

export type Item = Record<string, unknown>;

export const SIZES = [
	{ tag: '1440', width: 1440, height: 900, column: 380 },
	{ tag: '390', width: 390, height: 844, column: 358 }
];

export type Size = (typeof SIZES)[number];

export function imageItem(n: number, extra: Item = {}): Item {
	return {
		path: `uploads/e2e-image-${n}.png`,
		relative_path: `uploads/e2e-image-${n}.png`,
		url: `/api/e2e-media/image-${n}.png`,
		name: `harbour-${n}.png`,
		type: 'image',
		metadata: { width: 768, height: 512, size: 145408 },
		...extra
	};
}

export function videoItem(n: number): Item {
	return {
		path: `uploads/e2e-video-${n}.webm`,
		relative_path: `uploads/e2e-video-${n}.webm`,
		url: `/api/e2e-media/video-${n}.webm`,
		name: `clip-${n}.webm`,
		type: 'video',
		metadata: { width: 1280, height: 720, duration_seconds: 12, fps: 24, size: 8808038 }
	};
}

export function audioItem(n: number): Item {
	return {
		path: `uploads/e2e-audio-${n}.wav`,
		relative_path: `uploads/e2e-audio-${n}.wav`,
		url: `/api/e2e-media/audio-${n}.wav`,
		name: `voice-${n}.wav`,
		type: 'audio',
		metadata: { duration_seconds: 5, size: 225280 }
	};
}

export function generatedImageItem(): Item {
	return {
		path: 'outputs/2026-01-01/gen-1/0.png',
		relative_path: 'outputs/2026-01-01/gen-1/0.png',
		url: '/api/media/generations/gen-1/0.png',
		name: '0.png',
		type: 'image',
		metadata: { width: 768, height: 512, size: 145408 }
	};
}

export const UPLOADS = [
	{ id: 'row-image-1', filename: 'e2e-image-1.png' },
	{ id: 'row-video-1', filename: 'e2e-video-1.webm' },
	{ id: 'row-audio-1', filename: 'e2e-audio-1.wav' }
];

export async function prepareFieldPage(
	page: Page,
	size: Size,
	options: MediaToolMockOptions = {}
): Promise<PreparedLog> {
	await page.setViewportSize({ width: size.width, height: size.height });
	const log = await mockMediaTools(page, { uploads: UPLOADS, generationFiles: { 'gen-1': [{ id: 'file-1', filename: '0.png' }] }, ...options });
	await loginAsOwner(page);
	await page.goto('/history');
	await page.waitForFunction(() => !!(window as any).__potionui?.components?.MediaLoaderField);

	const webm = await recordWebm(page);
	const wav = fixtureWav(5);
	const png = solidPng(768, 512);
	await page.route('**/e2e-media/**', async (route) => {
		const url = route.request().url();
		if (url.endsWith('.png')) return route.fulfill({ status: 200, contentType: 'image/png', body: png });
		if (url.endsWith('.wav')) return route.fulfill({ status: 200, contentType: 'audio/wav', body: wav });
		if (url.endsWith('.webm') && webm) return route.fulfill({ status: 200, contentType: 'video/webm', body: webm });
		return route.fulfill({ status: 404, body: 'not served' });
	});
	await page.route('**/api/media/generations/gen-1/0.png', (route) =>
		route.fulfill({ status: 200, contentType: 'image/png', body: png })
	);
	return Object.assign(log, { hasVideo: webm !== null });
}

export async function mountField(page: Page, size: Size, props: Record<string, unknown>): Promise<Locator> {
	await page.evaluate(
		({ props, column }) => {
			const w = window as any;
			if (w.__mf) {
				w.__mf.entry.unmount(w.__mf.handle);
				w.__mf.el.remove();
			}
			const entry = w.__potionui.components.MediaLoaderField;
			const el = document.createElement('div');
			el.id = 'mf-host';
			el.style.cssText = `position:fixed;top:16px;left:8px;width:${column}px;max-height:calc(100vh - 32px);overflow-y:auto;overflow-x:hidden;z-index:60;padding:12px;`;
			el.className = 'bg-surface-1 border border-line rounded-lg';
			document.body.appendChild(el);
			const state: any = { entry, el, published: [], masks: [] };
			state.handle = entry.mount(el, {
				name: 'refs',
				...props,
				onChange: (_name: string, value: unknown) => {
					state.published.push(value);
					entry.update(state.handle, { value });
				},
				onMaskChange: (_name: string, path: string | undefined) => state.masks.push(path ?? null)
			});
			w.__mf = state;
		},
		{ props, column: size.column }
	);
	return page.locator('#mf-host');
}

export async function published(page: Page): Promise<Item[]> {
	return page.evaluate(() => (window as any).__mf.published);
}

export async function pickTool(page: Page, host: Locator, id: string) {
	await host.locator('[data-tools-trigger]').first().click();
	const item = page.locator(`[data-tools-menu] [data-tool="${id}"]`);
	await item.waitFor({ state: 'visible' });
	await item.click();
}
