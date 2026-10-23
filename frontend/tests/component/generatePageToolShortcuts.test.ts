import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const source = readFileSync(resolve(process.cwd(), 'src/routes/generate/+page.svelte'), 'utf8');

function handlerBody(action: string): string {
	const start = source.indexOf(`keybindingsStore.registerHandler('${action}', () => {`);
	expect(start, `${action} handler`).toBeGreaterThan(-1);
	const end = source.indexOf('\n\t\t});', start);
	return source.slice(start, end);
}

describe('generate page tool shortcuts', () => {
	for (const action of ['toggle_left_panel', 'toggle_floating_form', 'toggle_floating_workbench']) {
		it(`${action} does nothing while the toolbar tools are hidden`, () => {
			const body = handlerBody(action).split('\n').slice(1).join('\n').trimStart();
			expect(body.startsWith('if (!contextToolsAvailable(currentTab)) return;')).toBe(true);
		});
	}

	it('leaves the shortcuts that have no toolbar button alone', () => {
		expect(handlerBody('toggle_workbench_panel')).not.toContain('contextToolsAvailable');
		expect(handlerBody('toggle_last_generations')).not.toContain('contextToolsAvailable');
	});
});
