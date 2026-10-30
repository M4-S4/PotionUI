import { chromium, type FullConfig } from '@playwright/test';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { AUTH_STATE_PATH, SIGNED_OUT, loginAsOwnerViaForm } from './helpers';

const LOGIN_ATTEMPTS = 3;

export default async function globalSetup(config: FullConfig) {
	if (process.env.E2E_UNCLAIMED) {
		mkdirSync(dirname(AUTH_STATE_PATH), { recursive: true });
		writeFileSync(AUTH_STATE_PATH, JSON.stringify(SIGNED_OUT));
		return () => rmSync(AUTH_STATE_PATH, { force: true });
	}
	const baseURL = config.projects[0].use.baseURL;
	const browser = await chromium.launch();
	try {
		const context = await browser.newContext({ baseURL });
		for (let attempt = 1; ; attempt++) {
			const page = await context.newPage();
			try {
				await loginAsOwnerViaForm(page);
				break;
			} catch (error) {
				await page.close();
				if (attempt === LOGIN_ATTEMPTS) throw error;
				console.warn(`[global-setup] owner sign-in attempt ${attempt} failed, retrying: ${error}`);
			}
		}
		mkdirSync(dirname(AUTH_STATE_PATH), { recursive: true });
		await context.storageState({ path: AUTH_STATE_PATH });
	} finally {
		await browser.close();
	}
	return () => rmSync(AUTH_STATE_PATH, { force: true });
}
