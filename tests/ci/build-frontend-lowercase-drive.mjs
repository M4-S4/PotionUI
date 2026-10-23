import { spawnSync } from 'node:child_process';

const command = process.argv.slice(2).join(' ');
if (!command) {
	console.error('usage: node build-frontend-lowercase-drive.mjs <command...>');
	process.exit(2);
}

const cwd = process.cwd().replace(/^[A-Za-z]:/, (drive) => drive.toLowerCase());

const probe = spawnSync('node -p "process.cwd()"', { cwd, shell: true, encoding: 'utf-8' });
const childCwd = probe.stdout.trim();
console.log(`child cwd: ${childCwd}`);
if (process.platform === 'win32' && !/^[a-z]:/.test(childCwd)) {
	console.error('the child process cwd does not start with a lowercase drive letter');
	process.exit(1);
}

const result = spawnSync(command, { cwd, shell: true, stdio: 'inherit' });
process.exit(result.status ?? 1);
