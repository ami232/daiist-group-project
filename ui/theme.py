"""Design tokens (Aerospace Blue identity) and the Gradio theme built from them.

The same values are declared as CSS variables in assets/styles.css (`--mai-*`).
Keep both in sync. Contrast ratios are WCAG 2.1, measured on 2026-11-25/26.
"""

import inspect

import gradio as gr

# Brand
DEEP_OCEAN = "#102E4A"  # header, headings, primary button (13.9:1 on white)
PETROL = "#007C9E"  # large accents, icons, charts (4.80 on white, 4.38 on Cloud: not for small text on Cloud)
PETROL_TEXT = "#006D8C"  # accent text and links (5.89 on white, 5.38 on Cloud)
ICE = "#A8D8EA"  # fills only; as text only on Deep Ocean (9.05)
STEEL = "#718096"  # borders and large labels only
STEEL_TEXT = "#5F6B7E"  # secondary text (5.40 on white, 4.93 on Cloud)
CLOUD = "#F1F5F9"  # page background
WHITE = "#FFFFFF"  # cards
INK = "#0F1E2E"  # body text
BORDER = "#DCE3EA"

FONT = [gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"]
FONT_MONO = ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"]


def build_theme() -> gr.themes.Base:
    """A neutral Gradio base so the native widgets (inputs, buttons, tabs) match the CSS design system."""
    theme = gr.themes.Base(font=FONT, font_mono=FONT_MONO, radius_size=gr.themes.sizes.radius_md,
                           text_size=gr.themes.sizes.text_md, spacing_size=gr.themes.sizes.spacing_md)
    light = dict(
        body_background_fill=CLOUD,
        body_text_color=INK,
        body_text_color_subdued=STEEL_TEXT,
        background_fill_primary=WHITE,
        background_fill_secondary=CLOUD,
        block_background_fill=WHITE,
        block_border_color=BORDER,
        block_border_width="1px",
        block_label_text_color=STEEL_TEXT,
        block_title_text_color=INK,
        block_info_text_color=STEEL_TEXT,
        block_shadow="none",
        border_color_primary=BORDER,
        border_color_accent=PETROL,
        color_accent=PETROL,
        color_accent_soft="#E3F2F8",
        link_text_color=PETROL_TEXT,
        link_text_color_hover=DEEP_OCEAN,
        input_background_fill=WHITE,
        input_border_color="#CBD5E1",
        input_border_width="1px",
        input_padding="9px 12px",
        input_radius="6px",
        input_border_color_focus=PETROL,
        input_shadow="none",
        input_shadow_focus="0 0 0 3px rgba(0, 124, 158, 0.18)",
        button_primary_background_fill=DEEP_OCEAN,
        button_primary_background_fill_hover="#1A4166",
        button_primary_text_color=WHITE,
        button_primary_border_color=DEEP_OCEAN,
        button_secondary_background_fill=WHITE,
        button_secondary_background_fill_hover="#F5F8FB",
        button_secondary_text_color=DEEP_OCEAN,
        button_secondary_border_color="#CBD5E1",
        checkbox_label_background_fill=WHITE,
        checkbox_label_background_fill_selected="#E3F2F8",
        checkbox_label_border_color="#CBD5E1",
        checkbox_label_border_color_selected=PETROL,
        checkbox_label_text_color_selected=DEEP_OCEAN,
        checkbox_background_color_selected=PETROL,
        checkbox_border_color_selected=PETROL,
        loader_color=PETROL,
        slider_color=PETROL,
    )
    # The design is light-only: dark-mode variables get the same values, so the
    # widgets never turn dark next to the white custom cards.
    accepted = inspect.signature(theme.set).parameters
    dark = {f"{key}_dark": value for key, value in light.items() if f"{key}_dark" in accepted}
    return theme.set(**light, **dark)
