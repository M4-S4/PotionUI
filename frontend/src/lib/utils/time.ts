export function startOfDay(date: Date): number {
	return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

export function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

export type Debounced<T extends (...args: any[]) => void> = T & { cancel: () => void };

export function debounce<T extends (...args: any[]) => void>(fn: T, ms: number): Debounced<T> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	const debounced = ((...args: Parameters<T>) => {
		clearTimeout(timer);
		timer = setTimeout(() => fn(...args), ms);
	}) as Debounced<T>;
	debounced.cancel = () => {
		clearTimeout(timer);
		timer = undefined;
	};
	return debounced;
}
