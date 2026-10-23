type Paths = string | string[];

const chat: Record<string, Paths> = {
	menu: 'M4 7h16M4 12h16M4 17h16',
	plus: 'M12 5v14M5 12h14',
	search: [
		'M4 11a7 7 0 1 0 14 0a7 7 0 1 0 -14 0',
		'm20 20-4-4'
	],
	edit: 'M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4Z',
	close: 'm6 6 12 12M18 6 6 18',
	more: [],
	chevron: 'm8 10 4 4 4-4',
	sparkles: 'm12 3 1.2 3.8L17 8l-3.8 1.2L12 13l-1.2-3.8L7 8l3.8-1.2ZM6 14l.8 2.2L9 17l-2.2.8L6 20l-.8-2.2L3 17l2.2-.8Zm12-1 1 2.8 2.8 1-2.8 1-1 2.8-1-2.8-2.8-1 2.8-1Z',
	image: [
		'M5 4h14a2 2 0 0 1 2 2v12a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2v-12a2 2 0 0 1 2 -2z',
		'M7 9a2 2 0 1 0 4 0a2 2 0 1 0 -4 0',
		'm21 15-4.5-4.5L6 20'
	],
	tools: [
		'm14.7 6.3 3-3a4 4 0 0 1-5 5l-6.9 6.9a2.1 2.1 0 0 1-3-3l6.9-6.9a4 4 0 0 1 5-5l-3 3Z',
		'm15 13 6 6-2 2-6-6'
	],
	memory: 'M9 5a3 3 0 0 0-5.4 1.8A3 3 0 0 0 3 12a3 3 0 0 0 1 5.8A3 3 0 0 0 9 20ZM15 5a3 3 0 0 1 5.4 1.8A3 3 0 0 1 21 12a3 3 0 0 1-1 5.8A3 3 0 0 1 15 20ZM9 3v18M15 3v18',
	'arrow-up': 'M12 19V5m-6 6 6-6 6 6',
	copy: [
		'M11 9h7a2 2 0 0 1 2 2v7a2 2 0 0 1 -2 2h-7a2 2 0 0 1 -2 -2v-7a2 2 0 0 1 2 -2z',
		'M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3'
	],
	good: 'M7 10v11H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2Zm0 9 2 2h7.5a2 2 0 0 0 1.9-1.4l2.3-7A2 2 0 0 0 18.8 10H14l.7-3.3A3 3 0 0 0 12 3L7 10',
	bad: 'M7 14V3H4a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2Zm0-9 2-2h7.5a2 2 0 0 1 1.9 1.4l2.3 7a2 2 0 0 1-1.9 2.6H14l.7 3.3A3 3 0 0 1 12 21l-5-7',
	refresh: [
		'M20 7v5h-5M4 17v-5h5',
		'M6.1 9a7 7 0 0 1 11.3-2L20 9M4 15l2.6 2a7 7 0 0 0 11.3-2'
	],
	check: 'm5 12 4 4L19 6',
	file: [
		'M6 3h8l4 4v14H6Z',
		'M14 3v5h5'
	],
	wand: [
		'm15 4 5 5L8 21l-5-5Z',
		'm6 13 5 5M6 3v3M4.5 4.5h3M17 16v4M15 18h4M17 2v3M15.5 3.5h3'
	],
	sliders: [
		'M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5',
		'M14 6a2 2 0 1 0 4 0a2 2 0 1 0 -4 0',
		'M6 12a2 2 0 1 0 4 0a2 2 0 1 0 -4 0',
		'M11 18a2 2 0 1 0 4 0a2 2 0 1 0 -4 0'
	],
	clock: [
		'M3 12a9 9 0 1 0 18 0a9 9 0 1 0 -18 0',
		'M12 7v5l3 2'
	],
	info: [
		'M3 12a9 9 0 1 0 18 0a9 9 0 1 0 -18 0',
		'M12 11v5M12 8h.01'
	],
	sun: [
		'M8 12a4 4 0 1 0 8 0a4 4 0 1 0 -8 0',
		'M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.66 6.34l1.41-1.41'
	],
	lock: [
		'M7 10h10a2 2 0 0 1 2 2v7a2 2 0 0 1 -2 2h-10a2 2 0 0 1 -2 -2v-7a2 2 0 0 1 2 -2z',
		'M8 10V7a4 4 0 0 1 8 0v3'
	],
	upload: 'M12 16V4m-5 5 5-5 5 5M5 20h14',
	library: [
		'M6 4h12a2 2 0 0 1 2 2v12a2 2 0 0 1 -2 2h-12a2 2 0 0 1 -2 -2v-12a2 2 0 0 1 2 -2z',
		'm4 15 4-4 3 3 3-3 6 6',
		'M14 8.5a1.5 1.5 0 1 0 3 0a1.5 1.5 0 1 0 -3 0'
	],
	paste: [
		'M9 5h6M9 3h6v4H9Z',
		'M9 5H6a2 2 0 0 0-2 2v13h11v-3M15 11h5v10h-8v-7Z'
	],
	pin: [
		'm14 4 6 6-3 1-4 4-1 4-7-7 4-1 4-4Z',
		'm9 16-5 5'
	],
	trash: 'M4 7h16M9 7V4h6v3M6 7l1 14h10l1-14M10 11v6M14 11v6'
};

const chatFilled: Record<string, string[]> = {
	more: ['M4 12a1 1 0 1 0 2 0a1 1 0 1 0 -2 0', 'M11 12a1 1 0 1 0 2 0a1 1 0 1 0 -2 0', 'M18 12a1 1 0 1 0 2 0a1 1 0 1 0 -2 0']
};

const composer: Record<string, Paths> = {
	library: 'M4 5h5v14H4ZM9 7h5v12H9ZM14 4h6v15h-6Z',
	sparkles: chat.sparkles,
	template: [
		'M5 4h14a2 2 0 0 1 2 2v12a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2v-12a2 2 0 0 1 2 -2z',
		'M3 10h18M9 10v10'
	],
	braces: 'M9 3H7a2 2 0 0 0-2 2v4l-2 3 2 3v4a2 2 0 0 0 2 2h2M15 3h2a2 2 0 0 1 2 2v4l2 3-2 3v4a2 2 0 0 1-2 2h-2',
	more: chat.more,
	grip: [],
	copy: chat.copy,
	'eye-off': 'm3 3 18 18M10.7 10.7a2 2 0 0 0 2.6 2.6M9.9 5.2A10 10 0 0 1 12 5c5 0 8.5 4.5 9.5 7a13 13 0 0 1-2.1 3.3M6.2 6.2A13 13 0 0 0 2.5 12c1 2.5 4.5 7 9.5 7 1.4 0 2.7-.3 3.8-.8',
	eye: [
		'M2.5 12C3.5 9.5 7 5 12 5s8.5 4.5 9.5 7c-1 2.5-4.5 7-9.5 7s-8.5-4.5-9.5-7Z',
		'M9.5 12a2.5 2.5 0 1 0 5 0a2.5 2.5 0 1 0 -5 0'
	],
	'chevron-down': 'm7 9 5 5 5-5',
	'chevron-up': 'm7 15 5-5 5 5',
	plus: chat.plus,
	info: chat.info,
	file: chat.file,
	check: chat.check,
	close: chat.close,
	sliders: chat.sliders,
	shuffle: 'M4 7h3c5 0 5 10 10 10h3M17 4l3 3-3 3M4 17h3c2 0 3.3-1.6 4.5-3.5M16 7h4M17 14l3 3-3 3',
	search: [
		'M4 11a7 7 0 1 0 14 0a7 7 0 1 0 -14 0',
		'm16 16 5 5'
	],
	folder: 'M3 6h7l2 2h9v11H3Z',
	pencil: [
		'm4 20 4-1 11-11-3-3L5 16Z',
		'm14 7 3 3'
	],
	save: [
		'M5 3h12l2 2v16H5Z',
		'M8 3v6h8V3M8 21v-7h8v7'
	],
	trash: 'M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14M10 11v6M14 11v6',
	pin: [
		'M12 2a5 5 0 0 0-5 5c0 3 2 5 3 7l2 6 2-6c1-2 3-4 3-7a5 5 0 0 0-5-5Z',
		'M10 7a2 2 0 1 0 4 0a2 2 0 1 0 -4 0'
	],
	'chevron-right': 'm9 5 7 7-7 7',
	image: [
		'M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2v-14a2 2 0 0 1 2 -2z',
		'M7 9a2 2 0 1 0 4 0a2 2 0 1 0 -4 0',
		'M21 15l-5-5L5 21'
	],
	play: [],
	code: 'M16 18l6-6-6-6M8 6l-6 6 6 6'
};

const composerFilled: Record<string, string[]> = {
	more: chatFilled.more,
	grip: ['M8 6a1 1 0 1 0 2 0a1 1 0 1 0 -2 0', 'M14 6a1 1 0 1 0 2 0a1 1 0 1 0 -2 0', 'M8 12a1 1 0 1 0 2 0a1 1 0 1 0 -2 0', 'M14 12a1 1 0 1 0 2 0a1 1 0 1 0 -2 0', 'M8 18a1 1 0 1 0 2 0a1 1 0 1 0 -2 0', 'M14 18a1 1 0 1 0 2 0a1 1 0 1 0 -2 0'],
	play: ['M6 4v16l14-8Z']
};

const panel: Record<string, Paths> = {
	x: chat.close,
	loop: [
		'M20 12a8 8 0 1 1-2.34-5.66',
		'M20 4v6h-6'
	],
	pause: 'M9 6v12M15 6v12',
	hourglass: 'M7 3h10M7 21h10M8 3c0 4 1.5 6 4 9-2.5 3-4 5-4 9M16 3c0 4-1.5 6-4 9 2.5 3 4 5 4 9',
	sliders: 'M4 7h5M15 7h5M4 17h9M17 17h3M9 4v6M13 14v6',
	history: [
		'M4 12a8 8 0 1 0 2.3-5.7L4 8.6',
		'M4 4v4.6h4.6M12 8v4l3 2'
	],
	session: 'M6 4h12v16l-6-3-6 3Z',
	save: [
		'M5 4h12l2 2v14H5Z',
		'M8 4v6h8V4M8 20v-6h8v6'
	],
	chevron: chat.chevron,
	'chevron-left': 'm15 18-6-6 6-6',
	plus: chat.plus,
	trash: 'M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5',
	arrow: 'M5 12h14M14 7l5 5-5 5',
	extension: 'M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z'
};

function prefixed<T>(prefix: string, glyphs: Record<string, T>): Record<string, T> {
	return Object.fromEntries(Object.entries(glyphs).map(([name, value]) => [`${prefix}-${name}`, value]));
}

export const spriteGlyphs: Record<string, Paths> = {
	...prefixed('chat', chat),
	...prefixed('composer', composer),
	...prefixed('panel', panel)
};

export const spriteFilledGlyphs: Record<string, string[]> = {
	...prefixed('chat', chatFilled),
	...prefixed('composer', composerFilled)
};
