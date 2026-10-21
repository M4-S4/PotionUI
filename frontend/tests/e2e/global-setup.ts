import { chromium, type FullConfig } from '@playwright/test';
import { mkdirSync, rmSync } from 'node:fs';
import { dirname } from 'node:path';
import { AUTH_STATE_PATH, loginAsOwnerViaForm } from './helpers';

export default async function globalSetup(config: FullConfig) {
	const baseURL = config.projects[0].use.baseURL;
	const browser = await chromium.launch();
	try {
		const context = await browser.newContext({ baseURL });
		const page = await context.newPage();
		await loginAsOwnerViaForm(page);
		mkdirSync(dirname(AUTH_STATE_PATH), { recursive: true });
		await context.storageState({ path: AUTH_STATE_PATH });
	} finally {
		await browser.close();
	}
	return () => rmSync(AUTH_STATE_PATH, { force: true });
}
