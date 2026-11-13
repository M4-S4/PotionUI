import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { newAsset } from '../../frontend/src/lib/starter.js';
import { gatherPicks } from '../../frontend/src/lib/kits.js';
import { stableIds } from './make_starter_lumi.mjs';

const here = dirname(fileURLToPath(import.meta.url));

export const SEED = 127;
export const COUNT = 40;
export const KINDS = ['image', 'video', 'audio', 'text'];
export const PLACES = [['image', 'video'], ['audio'], ['image'], ['video']];
const NAMES = ['face', 'face', 'hands', 'coat', 'street', 'voice', 'rain', 'theme'];

export function random(seed) {
	let state = seed >>> 0;
	return () => {
		state = (state + 0x6d2b79f5) >>> 0;
		let x = state;
		x = Math.imul(x ^ (x >>> 15), x | 1);
		x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
		return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
	};
}

const pick = (rng, list) => list[Math.floor(rng() * list.length)];

export const uploadName = (ref) => `${NAMES[Number(ref.split('-').pop()) % NAMES.length]}.${ref.startsWith('up-audio') ? 'wav' : ref.startsWith('up-video') ? 'mp4' : 'png'}`;

function kitItem(rng, kitId, index, position) {
	const kind = pick(rng, KINDS);
	const id = `it_${kitId.slice(3)}${String(index).padStart(2, '0')}`;
	const name = `${pick(rng, NAMES)}_${index}`;
	const base = { id, kit_id: kitId, kind, name, position, media_ref: null, text: null, url: null, thumbnail_url: null, width: null, height: null, duration_seconds: null, missing: false, created_at: '2026-11-13T00:00:00Z' };
	if (kind === 'text') return { ...base, text: `A line about ${name}.` };
	if (rng() < 0.15) return { ...base, missing: true };
	const ref = `up-${kind}-${Math.floor(rng() * 12)}`;
	return { ...base, media_ref: ref, url: `/api/media/uploads/${uploadName(ref)}` };
}

export function seededKits(seed = SEED, count = COUNT) {
	const rng = random(seed);
	const cases = [];
	for (let n = 0; n < count; n++) {
		const kitId = `kt_${String(n).padStart(8, '0')}`;
		const size = n === 0 ? 50 : n === 1 ? 0 : Math.floor(rng() * 51);
		const positions = Array.from({ length: size }, (_, i) => i);
		for (let i = positions.length - 1; i > 0; i--) {
			const j = Math.floor(rng() * (i + 1));
			[positions[i], positions[j]] = [positions[j], positions[i]];
		}
		const items = positions.map((position, index) => kitItem(rng, kitId, index, position));
		const counts = Object.fromEntries(KINDS.map((kind) => [kind, items.filter((item) => item.kind === kind).length]));
		const kit = { id: kitId, name: n % 3 ? `Kit ${n}` : 'Mara', handle: `kit_${n}`, label: null, position: n, item_count: size, counts, preview_items: [], created_at: '2026-11-13T00:00:00Z', updated_at: '2026-11-13T00:00:00Z', items };
		cases.push({ kit, allowed: pick(rng, PLACES) });
	}
	return cases;
}

export function mediaOf(origin) {
	const ref = origin.itemId;
	return { mediaType: ref.split('-')[1], url: `/api/media/uploads/${uploadName(ref)}`, filename: uploadName(ref) };
}

export function assetsOf(picks) {
	return picks.map((p) => {
		const media = mediaOf(p.origin);
		return newAsset({ kind: media.mediaType, name: media.filename, src: media.url, origin: p.origin });
	});
}

export function makeKitPicks() {
	const cases = seededKits().map(({ kit, allowed }) => {
		const { picks, left } = gatherPicks({ kits: [kit] }, allowed);
		return { kit, allowed, left, assets: assetsOf(picks) };
	});
	return stableIds(JSON.parse(JSON.stringify({ seed: SEED, cases })));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
	writeFileSync(join(here, 'kit_picks.json'), JSON.stringify(makeKitPicks(), null, 2) + '\n');
}
