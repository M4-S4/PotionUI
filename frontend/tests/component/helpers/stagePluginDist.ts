import { copyFileSync, mkdirSync, mkdtempSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const STAGE_ROOT = resolve(__dirname, '../../../node_modules/.plugin-dist-under-test');

export function stagePluginDist(distFile: string): string {
	mkdirSync(STAGE_ROOT, { recursive: true });
	const stageDir = mkdtempSync(join(STAGE_ROOT, 'run-'));
	const distDir = dirname(distFile);
	const siblings = readdirSync(distDir).filter((name) => /^chunk-.*\.js$/.test(name));
	for (const name of siblings) {
		copyFileSync(join(distDir, name), join(stageDir, name));
	}
	const staged = join(stageDir, distFile.slice(distDir.length + 1).replace(/^[\\/]+/, ''));
	copyFileSync(distFile, staged);
	return pathToFileURL(staged).href;
}
