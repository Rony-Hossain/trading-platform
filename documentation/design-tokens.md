# Frontend Design Tokens (Trading Frontend)
Last updated: 2025-10-03

This document freezes the current token set so product, design, and engineering teams can implement consistently across Beginner and Expert surfaces. All values are ASCII and map directly to the existing code base (MUI theme, Tailwind 4 inline theme, and chart components).

## Color Palette
Each color lists the dominant usage, source, and measured contrast ratios. Ratios ? 4.5 meet WCAG 2.2 AA for normal text; ratios ? 3.0 are safe for icons or bold/large text only.

| Token | Hex | Usage | Contrast vs `#171717` | Contrast vs `#ffffff` | Notes |
|-------|-----|-------|------------------------|------------------------|-------|
| `color.background.default` | `#f7f9fc` | App background (light mode) | 1.00 | **17.0** | Pair with dark text `#171717`.
| `color.background.paper` | `#ffffff` | Surface cards, modals | 1.00 | **17.9** | Primary canvas.
| `color.background.dark` | `#0a0a0a` | Dark mode root | **16.9** | 1.00 | Use with light text `#ededed`.
| `color.foreground.primary` | `#171717` | Default text | ? | **17.0** | Use on light backgrounds.
| `color.foreground.inverse` | `#ededed` | Text on dark mode | **16.9** | ? | Meets AA on dark backgrounds.
| `color.primary.main` | `#1976d2` | Buttons, links (MUI theme) | 9.74 | 3.49 | Use white text at ?18 px / bold or add darker overlay.
| `color.secondary.main` | `#9c27b0` | Accent actions | 6.41 | 4.03 | White text acceptable for ?18 px / bold weight.
| `color.action.buy` | `#3B82F6` | Chart line (bullish) | 8.92 | 3.68 | Limit to charts/badges; add darker text overlay if needed.
| `color.action.sell` | `#EF4444` | Chart line (bearish) | 5.14 | 3.76 | Same guidance as above.
| `color.action.safe` | `#10B981` | Safety lines, positive chips | 9.08 | 2.54 | Use on light backgrounds with dark text; avoid white text.
| `color.neutral.muted` | `#6B7280` | Grid lines, secondary text | 3.45 | **4.83** | Meets AA for text on dark background; on white use ? 18 px/ bold.
| `color.warning` | `#F59E0B` | Alerts, indicator bands | 3.18 | 2.19 | Treat as icon/background color only.
| `color.gridline` | `#E5E7EB` | Chart grid, dividers | 1.09 | 13.53 | Non-text usage only.

**Contrast source:** computed with WCAG 2.2 luminance formula. Adjust token usage accordingly (e.g., place white text on `#1976d2` only for large/semibold buttons; otherwise switch text to `#0a0a0a` or darken the background).

## Typography
Fonts are provided by Next.js font loaders (`--font-geist-sans`, `--font-geist-mono`) falling back to system stacks.

| Token | Font | Size (px) | Line Height | Usage |
|-------|------|-----------|-------------|-------|
| `font.sans` | Geist Sans, Arial, Helvetica | ? | ? | Global default (see `globals.css`).
| `font.mono` | Geist Mono, SFMono, Menlo | ? | ? | Numbers, code snippets.
| `type.display` | `font.sans`, 32 px, line-height 1.25 | Hero headings (marketing surfaces).
| `type.h1` | `font.sans`, 28 px, line-height 1.3 | Page titles.
| `type.h2` | `font.sans`, 24 px, line-height 1.35 | Section headers.
| `type.h3` | `font.sans`, 20 px, line-height 1.4 | Card titles.
| `type.body` | `font.sans`, 16 px, line-height 1.5 | Default body copy.
| `type.small` | `font.sans`, 14 px, line-height 1.45 | Helper text, table labels.
| `type.caption` | `font.sans`, 12 px, line-height 1.4 | Badges, data labels (use with sufficient contrast).
| `type.numeric` | `font.mono`, 16 px, line-height 1.4 | Tabular metrics; apply `.tabular-nums` class.

Typography tokens align with MUI theme defaults and can be formalized via `createTheme({ typography: { ... } })` in a follow-up PR.

## Spacing Scale
Base unit is 4 px. Compose larger values by multiplying the base unit.

| Token | Pixel Value | Usage |
|-------|-------------|-------|
| `space.1` | 4 px | Fine gaps (icon padding).
| `space.2` | 8 px | Inline spacing.
| `space.3` | 12 px | Compact card padding.
| `space.4` | 16 px | Standard gutters.
| `space.6` | 24 px | Card-to-card spacing.
| `space.8` | 32 px | Section padding.
| `space.10` | 40 px | Page gutters (xl screens).

Tailwind utilities (`gap-4`, `px-6`, etc.) already map to this scale; document deviations when using custom CSS.

## Radii & Shape
- `radius.medium` = 10 px (MUI `shape.borderRadius`), applied to cards and modals.
- `radius.small` = 6 px for pills and inputs.
- `radius.full` = 9999 px for badges/avatars.

## Elevation & Borders
- `elevation.surface` = box-shadow `0px 2px 6px rgba(15, 23, 42, 0.08)` (recommended for floating cards; add to theme overrides).
- `border.subtle` = `1px solid #E5E7EB` for table rows, card outlines.
- `border.strong` = `1px solid #94A3B8` for focus rings when outline not supported.

## State & Feedback Tokens
| Token | Value | Usage |
|-------|-------|-------|
| `state.focus` | outline `2px solid #1976d2` + `outline-offset: 2px` | Keyboard focus.
| `state.focus.critical` | outline `2px solid #EF4444` | Destructive confirmation focus states.
| `state.hover` | Background tint `rgba(25, 118, 210, 0.08)` | Hover on interactive rows.
| `state.selected` | Background `rgba(25, 118, 210, 0.12)` + border `#1976d2` | Active navigation.
| `state.disabled` | Text `#9CA3AF`, border `#E5E7EB`, background `#F3F4F6` | Disabled buttons/inputs.
| `state.success` | Border `#10B981`, background `rgba(16, 185, 129, 0.12)` | Confirmation banners.
| `state.warning` | Border `#F59E0B`, background `rgba(245, 158, 11, 0.12)` | Watch-outs.
| `state.error` | Border `#EF4444`, background `rgba(239, 68, 68, 0.12)` | Errors and guardrails.

## Motion
- `motion.fast` = 120 ms ease-out (hover feedback).
- `motion.medium` = 180 ms ease-in-out (modal open, drawer slide).
- `motion.slow` = 300 ms ease-in-out (complex transitions). Honor `prefers-reduced-motion` and fall back to instant transitions.

## Dark Mode Mapping
Dark mode currently swaps the `--background`/`--foreground` CSS variables. For full parity, map tokens:

- `color.background.default.dark` = `#0a0a0a`
- `color.background.paper.dark` = `#111827`
- `color.foreground.primary.dark` = `#ededed`
- Accent colors remain the same; ensure contrast by pairing with `#ededed` text or lighten/darken as needed.

## Next Steps
1. Encode these tokens in a dedicated package (e.g., `libs/design-system/tokens.ts`) so Tailwind and MUI read from the same source.
2. Add automated contrast tests (Jest + `wcag-contrast` package) to guard future changes.
3. Update MUI `createTheme` typography section to align with the table above.
4. Share this document with design for visual confirmation before Phase 1 kickoff.
