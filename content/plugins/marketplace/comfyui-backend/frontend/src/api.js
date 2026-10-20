export async function call(url, init) {
	const res = await window.__potionui.api.fetch(url, init);
	const payload = await res.json().catch(() => null);
	return { ok: res.ok, status: res.status, payload };
}

export function messageOf(payload, fallback) {
	return payload?.detail || payload?.message || fallback;
}
