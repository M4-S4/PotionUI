import { spawnSync } from 'node:child_process';
import { realpathSync } from 'node:fs';

const RERUN_MARKER = 'POTIONUI_CANONICAL_CWD';

export interface CwdHost {
	platform: string;
	env: Record<string, string | undefined>;
	cwd: () => string;
	realpath: (path: string) => string;
	execPath: string;
	execArgv: string[];
	argv: string[];
	rerun: (command: string, args: string[], cwd: string, env: Record<string, string | undefined>) => number;
	exit: (code: number) => void;
}

const processHost: CwdHost = {
	platform: process.platform,
	env: process.env,
	cwd: () => process.cwd(),
	realpath: (path) => realpathSync.native(path),
	execPath: process.execPath,
	execArgv: process.execArgv,
	argv: process.argv,
	rerun: (command, args, cwd, env) => spawnSync(command, args, { cwd, env, stdio: 'inherit' }).status ?? 1,
	exit: (code) => process.exit(code)
};

function canonical(host: CwdHost, path: string): string {
	try {
		return host.realpath(path);
	} catch {
		return path;
	}
}

export function rerunFromCanonicalCwd(host: CwdHost = processHost): void {
	if (host.platform !== 'win32' || host.env[RERUN_MARKER]) return;
	const current = host.cwd();
	const target = canonical(host, current);
	if (target === current || target.toLowerCase() !== current.toLowerCase()) return;
	const [, script, ...rest] = host.argv;
	const args = [...host.execArgv, canonical(host, script), ...rest];
	host.exit(host.rerun(host.execPath, args, target, { ...host.env, [RERUN_MARKER]: '1' }));
}
