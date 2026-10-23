export function toneOf(percent) {
    if (percent < 50) return 'success';
    if (percent < 80) return 'warning';
    return 'danger';
}

export function accentOf(tone) {
    return `rgb(var(--${tone}))`;
}

export function usageLabelOf(percent) {
    if (percent < 50) return 'Plenty of headroom';
    if (percent < 80) return 'Moderate load';
    return 'Running near capacity';
}
