export const page = $state({
	url: new URL('http://localhost/generate'),
	state: {} as Record<string, unknown>
});

export function setPageState(url: URL, state: Record<string, unknown>) {
	page.url = url;
	page.state = state;
}
