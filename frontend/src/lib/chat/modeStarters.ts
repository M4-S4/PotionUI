export interface ChatStarter {
	title: string;
	sub: string;
	prompt: string;
	icon: string;
}

const GENERATION: ChatStarter[] = [
	{
		title: 'Explore art directions',
		sub: 'Three distinct approaches from the current image',
		prompt: 'Give me three art directions for the current image.',
		icon: 'i-sparkles'
	},
	{
		title: 'Review my settings',
		sub: 'Find quality, speed, or composition improvements',
		prompt: 'Review my current settings and suggest improvements.',
		icon: 'i-sliders'
	},
	{
		title: 'Refine this prompt',
		sub: 'Improve composition, detail, and lighting',
		prompt: 'Rewrite my prompt with stronger composition and lighting.',
		icon: 'i-wand'
	}
];

const HISTORY: ChatStarter[] = [
	{
		title: 'Find generations',
		sub: 'Search past work by subject, model, or date',
		prompt: 'Find my favorite generations from the past week.',
		icon: 'i-search'
	},
	{
		title: 'Review recent work',
		sub: 'See what you made and which settings worked',
		prompt: 'Summarize my recent generations and which settings gave the best results.',
		icon: 'i-image'
	},
	{
		title: 'Tag and organize',
		sub: 'Tidy untagged generations into groups',
		prompt: 'Suggest tags and collections for my untagged generations.',
		icon: 'i-library'
	}
];

const PROMPTS: ChatStarter[] = [
	{
		title: 'Find a saved prompt',
		sub: 'Search your library by subject or style',
		prompt: 'Search my saved prompts for portrait ideas.',
		icon: 'i-search'
	},
	{
		title: 'Save a new prompt',
		sub: 'Turn an idea into a reusable library entry',
		prompt: 'Write a detailed prompt for a candlelit potion shop and save it to my library.',
		icon: 'i-plus'
	},
	{
		title: 'Improve a saved prompt',
		sub: 'Sharpen composition, detail, and lighting',
		prompt: 'Pick one of my saved prompts and rewrite it with stronger composition and lighting.',
		icon: 'i-wand'
	}
];

const MODELS: ChatStarter[] = [
	{
		title: 'What do I have installed?',
		sub: 'A quick tour of your checkpoints and LoRAs',
		prompt: 'List my installed checkpoints and what each one is best at.',
		icon: 'i-library'
	},
	{
		title: 'Look up a model',
		sub: 'Type, base model, and where it is used',
		prompt: 'Tell me about my newest model: its type, base model, and recommended settings.',
		icon: 'i-info'
	},
	{
		title: 'Find the right model',
		sub: 'Match a model to what you want to make',
		prompt: 'Which of my models is best for photorealistic portraits?',
		icon: 'i-search'
	}
];

const PHRASEBOOK: ChatStarter[] = [
	{
		title: 'Browse categories',
		sub: 'See what your phrasebook already covers',
		prompt: 'List my phrasebook categories and how many values each one has.',
		icon: 'i-library'
	},
	{
		title: 'Add values',
		sub: 'Grow a category with fresh options',
		prompt: 'Suggest ten new values for one of my phrasebook categories.',
		icon: 'i-plus'
	},
	{
		title: 'Tidy a category',
		sub: 'Find duplicates and weak entries',
		prompt: 'Review one of my phrasebook categories for duplicates and weak values.',
		icon: 'i-sliders'
	}
];

const GENERIC: ChatStarter[] = [
	{
		title: 'What can you do here?',
		sub: 'A quick overview of this assistant',
		prompt: 'What can you help me with on this page?',
		icon: 'i-info'
	},
	{
		title: 'Summarize this page',
		sub: 'What is in front of you right now',
		prompt: 'Summarize what I am looking at on this page.',
		icon: 'i-search'
	},
	{
		title: 'Suggest a next step',
		sub: 'Where to go from here',
		prompt: 'Suggest a good next step for me.',
		icon: 'i-sparkles'
	}
];

const BY_MODE: Record<string, ChatStarter[]> = {
	generation: GENERATION,
	history: HISTORY,
	prompts: PROMPTS,
	models: MODELS,
	phrasebook: PHRASEBOOK
};

export function startersForMode(modeId: string | null | undefined): ChatStarter[] {
	return (modeId && BY_MODE[modeId]) || GENERIC;
}
