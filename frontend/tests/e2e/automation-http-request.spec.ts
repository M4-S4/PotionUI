import { test, expect } from '@playwright/test';
import { apiPost, loginAsOwner, ownerToken, screenshot } from './helpers';

const JOURNEY = 'automation-http-request';

test('a manual automation calls the backend over HTTP, passes the body to a later node, and the node can be tested from the editor', async ({ page }) => {
	test.setTimeout(120000);
	const backend = process.env.E2E_BACKEND_URL;
	test.skip(!backend, 'E2E_BACKEND_URL is not set');
	await page.setViewportSize({ width: 1440, height: 960 });
	await loginAsOwner(page);
	const token = await ownerToken(page);
	const headers = { Authorization: `Bearer ${token}` };

	const created = await apiPost(page, '/api/automations', token, {
		name: 'E2E HTTP request',
		graph: {
			nodes: [
				{ id: 'start', type: 'trigger.manual', position: { x: 0, y: 0 }, config: {} },
				{
					id: 'call',
					type: 'action.http_request',
					position: { x: 280, y: 0 },
					config: {
						method: 'GET',
						url: `${backend}/health`,
						headers: [{ name: 'Authorization', value: 'Bearer e2e-secret-token' }],
						allow_private_targets: true
					}
				},
				{
					id: 'after',
					type: 'action.send_notification',
					position: { x: 560, y: 0 },
					config: { title: 'HTTP result', message: 'service={{ upstream.call.body.service }}', level: 'info' }
				}
			],
			edges: [
				{ id: 'e1', source: 'start', source_handle: 'out', target: 'call', target_handle: 'in' },
				{ id: 'e2', source: 'call', source_handle: 'out', target: 'after', target_handle: 'in' }
			]
		}
	});
	const automationId: string = created.data.id;
	expect(JSON.stringify(created)).not.toContain('e2e-secret-token');

	const run = await page.request.post(`/api/automations/${automationId}/run`, { headers, data: {} });
	expect(run.ok()).toBeTruthy();
	const runId: string = (await run.json()).data.run_id;

	await expect
		.poll(
			async () => {
				const detail = await page.request.get(`/api/automations/${automationId}/runs/${runId}`, { headers });
				return (await detail.json()).data?.status;
			},
			{ timeout: 30000 }
		)
		.toBe('success');

	const detail = (await (await page.request.get(`/api/automations/${automationId}/runs/${runId}`, { headers })).json()).data;
	const call = detail.nodes.find((n: { node_id: string }) => n.node_id === 'call');
	const after = detail.nodes.find((n: { node_id: string }) => n.node_id === 'after');
	expect(call.output.body.service).toBe('potionui-api');
	expect(JSON.stringify(detail)).not.toContain('e2e-secret-token');
	expect(after.output.message).toBe('service=potionui-api');

	await page.goto(`/admin?tab=automations&id=${automationId}`);
	await page.locator('.svelte-flow__node').filter({ hasText: 'HTTP Request' }).first().click();
	await page.getByRole('button', { name: 'Test request', exact: true }).click();
	const result = page.locator('[data-http-test-result]');
	await expect(result).toBeVisible({ timeout: 15000 });
	await expect(result).toContainText('200');
	await expect(page.locator('[data-http-test-body]')).toContainText('potionui-api');
	await screenshot(page, JOURNEY, '1-test-request');
});
