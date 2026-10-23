export interface CommunityLink {
	id: 'website' | 'github' | 'reddit' | 'discord';
	label: string;
	href: string;
	icon: string;
}

export const COMMUNITY_LINKS: readonly CommunityLink[] = [
	{ id: 'website', label: 'Website', href: 'https://potionui.com', icon: 'globe' },
	{ id: 'github', label: 'GitHub', href: 'https://github.com/PotionUI/PotionUI', icon: 'github' },
	{ id: 'reddit', label: 'Reddit', href: 'https://www.reddit.com/r/PotionUI/', icon: 'reddit' },
	{ id: 'discord', label: 'Discord', href: 'https://discord.gg/avR4trp3b8', icon: 'discord' }
];
