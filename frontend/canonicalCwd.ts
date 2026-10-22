import { realpathSync } from 'node:fs';

export interface CwdHost {
	platform: string;
	cwd: () => string;
	chdir: (dir: string) => void;
	realpath: (dir: string) => string;
}

const processHost: CwdHost = {
	platform: process.platform,
	cwd: () => process.cwd(),
	chdir: (dir) => process.chdir(dir),
	realpath: (dir) => realpathSync.native(dir)
};

export function enterCanonicalCwd(host: CwdHost = processHost): void {
	if (host.platform !== 'win32') return;
	const current = host.cwd();
	let canonical: string;
	try {
		canonical = host.realpath(current);
	} catch {
		return;
	}
	if (canonical !== current && canonical.toLowerCase() === current.toLowerCase()) {
		host.chdir(canonical);
	}
}
