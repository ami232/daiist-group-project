"""A small, consistent inline SVG icon set (24 px grid, 1.75 px stroke, round caps).

Drawn for MaintAI. Every icon is decorative (`aria-hidden`): the text next to it
carries the meaning.
"""

_PATHS = {
    "pulse": '<path d="M3 12h4l2.5-6 4 12 2.5-6H21"/>',
    "clipboard": '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4.5h6M9 10h6M9 14h6M9 18h3"/>',
    "coins": '<ellipse cx="12" cy="6.5" rx="7" ry="2.5"/><path d="M5 6.5v5c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-5M5 11.5v5c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-5"/>',
    "trace": '<circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/><path d="M8 6h3a3 3 0 0 1 3 3v1M8 18h3a3 3 0 0 0 3-3v-1M14 12h2"/>',
    "sliders": '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16 9.8"/>',
    "alert": '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17.2v.1"/>',
    "block": '<circle cx="12" cy="12" r="9"/><path d="M5.7 5.7l12.6 12.6"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.8v.1"/>',
    "flask": '<path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4a2 2 0 0 0 1.8-3l-5-9V3"/><path d="M7.5 15h9"/>',
    "user-check": '<circle cx="10" cy="8" r="3.5"/><path d="M3.5 20a6.5 6.5 0 0 1 11.3-4.4M15 18l2 2 4-4"/>',
    "database": '<ellipse cx="12" cy="6" rx="7" ry="2.5"/><path d="M5 6v12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5V6M5 12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5"/>',
    "monitor": '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4M7 12l3-3 2 2 4-4"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "thumb-up": '<path d="M7 11v9H4v-9zM7 11l4-7a2 2 0 0 1 3 2l-1 4h5a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 16.8 20H7"/>',
    "thumb-down": '<path d="M7 13V4H4v9zM7 13l4 7a2 2 0 0 0 3-2l-1-4h5a2 2 0 0 0 2-2.3l-1.2-6A2 2 0 0 0 16.8 4H7"/>',
    "bell": '<path d="M6 16V11a6 6 0 0 1 12 0v5l1.5 2h-15z"/><path d="M10 20.5a2 2 0 0 0 4 0"/>',
    "shield": '<path d="M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6z"/><path d="M9 12l2 2 4-4"/>',
    "requests": '<path d="M4 6h16M4 12h10M4 18h7"/><circle cx="18" cy="16" r="3"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2"/>',
    "drift": '<path d="M3 17c3 0 3-6 6-6s3 4 6 4 3-8 6-8"/><path d="M3 21h18"/>',
    "wrench":'<path d="M14.5 5.5a4 4 0 0 0 4.9 4.9L20 11l-9 9a2.1 2.1 0 0 1-3-3l9-9 .6.6a4 4 0 0 0-3.1-3.1z"/>',
}


def icon(name: str, size: int = 18, cls: str = "mai-icon") -> str:
    return (f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true" focusable="false">{_PATHS[name]}</svg>')


# The brand mark (same drawing as assets/favicon.svg), for the header.
MARK = ('<svg class="mai-mark" width="32" height="32" viewBox="0 0 40 40" aria-hidden="true" focusable="false">'
        '<rect width="40" height="40" rx="9" fill="#0B2238"/>'
        '<path d="M6 27h4.6L14.6 13l5.4 11 5.4-11 4 14H34" fill="none" stroke="#FFFFFF" stroke-width="2.8" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="25.4" cy="13" r="4.6" fill="#0B2238"/><circle cx="25.4" cy="13" r="3" fill="#A8D8EA"/></svg>')
