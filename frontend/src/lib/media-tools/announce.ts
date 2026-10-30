export interface AnnounceOptions {
	title?: string;
	action?: { label: string; href: string };
	quietSuccess?: boolean;
	retry?: () => void;
	label?: string;
}

export type AnnounceSetting = boolean | 'failures' | AnnounceOptions;

export const SAVED_TO_LIBRARY_ANNOUNCE: AnnounceOptions = {
	title: 'Saved to Library',
	action: { label: 'Open Library', href: '/library' }
};
