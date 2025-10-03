# Accessibility Targets & Standards
Last updated: 2025-10-03

## Compliance Level

**Target**: WCAG 2.2 Level AA compliance across all surfaces

## Core Principles (POUR)

1. **Perceivable** - Information must be presentable to users in ways they can perceive
2. **Operable** - UI components must be operable by all users
3. **Understandable** - Information and UI operation must be understandable
4. **Robust** - Content must be robust enough to work with assistive technologies

## Color & Contrast

### Contrast Ratios (WCAG 2.2 AA)

- **Normal text (< 18pt)**: Minimum 4.5:1 contrast ratio
- **Large text (≥ 18pt or ≥ 14pt bold)**: Minimum 3:1 contrast ratio
- **UI components & graphical objects**: Minimum 3:1 contrast ratio
- **Focus indicators**: Minimum 3:1 contrast ratio against adjacent colors

### Verified Combinations

All color combinations in `design-tokens.md` have been validated. See token documentation for specific contrast ratios.

### Color Blindness Support

- **Never use color alone** to convey information
- **Patterns & labels** - Use icons, patterns, or text labels alongside color
- **Tested palettes** - Chart colors tested with deuteranopia, protanopia, and tritanopia simulators

Example chart color strategy:
```typescript
// Use distinct shapes + colors for data series
const chartConfig = {
  buy: { color: '#3B82F6', shape: 'circle', pattern: 'solid' },
  sell: { color: '#EF4444', shape: 'square', pattern: 'dashed' },
  hold: { color: '#10B981', shape: 'triangle', pattern: 'dotted' },
}
```

## Keyboard Navigation

### Tab Order & Focus Management

1. **Logical tab order** - Follow visual reading order (left-to-right, top-to-bottom)
2. **Skip links** - "Skip to main content" link as first focusable element
3. **Focus trap** - Modals and drawers trap focus until dismissed
4. **Focus restoration** - Return focus to trigger element after modal closes

### Keyboard Shortcuts

| Action | Shortcut | Context |
|--------|----------|---------|
| Navigate to Today | `Alt+T` | Global |
| Navigate to Portfolio | `Alt+P` | Global |
| Navigate to Alerts | `Alt+A` | Global |
| Navigate to Settings | `Alt+S` | Global |
| Open command palette | `Cmd+K` / `Ctrl+K` | Global |
| Refresh current view | `R` | When focused on main content |
| Open help | `?` | Global |
| Close modal/drawer | `Esc` | When modal is open |
| Confirm action | `Enter` | In dialogs |
| Cancel action | `Esc` | In dialogs |
| Navigate table rows | `↑` `↓` | In data tables |
| Expand/collapse row | `Space` | On expandable rows |

### Focus Indicators

- **Visible focus** - 2px solid outline with 2px offset
- **High contrast** - Primary color (`#1976d2`) for standard, red (`#EF4444`) for destructive actions
- **Never remove** - `:focus-visible` for keyboard users only, but never completely hidden

```css
/* Focus styles */
:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
  border-radius: 4px;
}

/* Destructive action focus */
.destructive:focus-visible {
  outline-color: var(--color-error);
}
```

## Screen Reader Support

### ARIA Labels & Roles

1. **Semantic HTML first** - Use native elements (`<button>`, `<nav>`, `<main>`) before ARIA
2. **ARIA when needed** - Add ARIA only when semantic HTML is insufficient
3. **Landmark regions** - Every page has `<main>`, `<nav>`, `<aside>` where appropriate
4. **Live regions** - Use `aria-live` for dynamic content updates

### Required ARIA Patterns

| Component | ARIA Pattern |
|-----------|--------------|
| Modal Dialog | `role="dialog"`, `aria-modal="true"`, `aria-labelledby` |
| Drawer/Sheet | `role="complementary"` or `role="dialog"` |
| Alert Banner | `role="alert"` (assertive) or `role="status"` (polite) |
| Data Table | Native `<table>` with `<caption>`, `<th scope>` |
| Tabs | `role="tablist"`, `role="tab"`, `role="tabpanel"` |
| Dropdown Menu | `role="menu"`, `role="menuitem"` |
| Tooltip | `role="tooltip"`, `aria-describedby` |
| Chart | `role="img"`, `aria-label`, data table alternative |

### Live Regions for Trading Data

```typescript
// Alert notifications - assertive (interrupts)
<div role="alert" aria-live="assertive">
  New opportunity: AAPL at $185
</div>

// Price updates - polite (doesn't interrupt)
<div role="status" aria-live="polite" aria-atomic="true">
  AAPL: $185.32 (+2.3%)
</div>

// Plan updates - polite
<div role="status" aria-live="polite">
  Plan updated 2 minutes ago
</div>
```

### Screen Reader Announcements

Use `aria-label` for icon-only buttons:

```tsx
<button aria-label="Refresh plan">
  <RefreshIcon />
</button>

<button aria-label="Close alert drawer">
  <CloseIcon />
</button>
```

For complex components, provide context:

```tsx
<div
  role="region"
  aria-label="Today's trading plan"
  aria-describedby="plan-description"
>
  <p id="plan-description" className="sr-only">
    List of recommended trades for today based on your risk profile and market conditions
  </p>
  {/* Plan content */}
</div>
```

## Reduced Motion

### Honor User Preferences

```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

### Alternative Feedback

When motion is reduced, provide alternative feedback:

```typescript
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

// Instead of animated spinner
{prefersReducedMotion ? (
  <div aria-busy="true">Loading...</div>
) : (
  <Spinner aria-label="Loading" />
)}

// Instead of slide animation
const drawerVariants = prefersReducedMotion
  ? { open: { opacity: 1 }, closed: { opacity: 0 } }
  : { open: { x: 0 }, closed: { x: 300 } }
```

## Forms & Input

### Form Labels

- **Always visible** - No placeholder-only labels
- **Associated** - Use `<label for>` or wrap input with `<label>`
- **Required indicators** - Both visual (`*`) and `aria-required="true"`
- **Error messages** - Linked with `aria-describedby` and `aria-invalid`

```tsx
<div className="form-field">
  <label htmlFor="loss-cap">
    Daily Loss Cap (%) <span aria-label="required">*</span>
  </label>
  <input
    id="loss-cap"
    type="number"
    aria-required="true"
    aria-invalid={hasError}
    aria-describedby={hasError ? "loss-cap-error" : undefined}
  />
  {hasError && (
    <span id="loss-cap-error" role="alert">
      Loss cap must be between 1% and 10%
    </span>
  )}
</div>
```

### Input Validation

- **Inline errors** - Show errors immediately below field
- **Error summary** - List all errors at top of form on submit
- **Success feedback** - Confirm successful saves with temporary message

## Charts & Data Visualization

### Text Alternatives

Every chart must provide:

1. **Accessible label** - `aria-label` describing chart purpose
2. **Data table fallback** - Hidden `<table>` with same data for screen readers
3. **Trend summary** - Text summary of key insights

```tsx
<div className="chart-container">
  <div role="img" aria-label="Price chart for AAPL showing upward trend over last 30 days">
    <Chart data={priceData} />
  </div>

  {/* Screen reader alternative */}
  <table className="sr-only" aria-label="AAPL price data">
    <caption>Price history for AAPL (last 30 days)</caption>
    <thead>
      <tr>
        <th scope="col">Date</th>
        <th scope="col">Price</th>
        <th scope="col">Change</th>
      </tr>
    </thead>
    <tbody>
      {priceData.map((row) => (
        <tr key={row.date}>
          <td>{row.date}</td>
          <td>{row.price}</td>
          <td>{row.change}</td>
        </tr>
      ))}
    </tbody>
  </table>

  {/* Trend summary */}
  <p className="sr-only">
    AAPL has increased 12% over the last 30 days, from $165 to $185, with moderate volatility.
  </p>
</div>
```

### Interactive Charts

For interactive charts (zoom, pan):
- Provide keyboard alternatives for mouse-only interactions
- Use `aria-live` to announce data point details on focus
- Offer data table view toggle

## Touch Targets

### Minimum Size (WCAG 2.2 AAA)

- **Target size**: Minimum 44×44 CSS pixels for all interactive elements
- **Spacing**: Minimum 8px between adjacent targets
- **Exception**: Inline text links can be smaller if sufficient spacing exists

```css
/* Ensure minimum touch target */
button, a, input[type="checkbox"], input[type="radio"] {
  min-width: 44px;
  min-height: 44px;
}

/* Add padding to small icons */
.icon-button {
  padding: 10px;
  min-width: 44px;
  min-height: 44px;
}
```

## Testing Requirements

### Automated Testing

1. **axe DevTools** - Run on every page before release
2. **Lighthouse** - Accessibility score ≥ 95
3. **WAVE** - Zero errors on critical paths
4. **Pa11y CI** - Automated checks in CI/CD pipeline

```bash
# Run accessibility audit
npm run test:a11y

# CI check
pa11y-ci --config .pa11yci.json
```

### Manual Testing

1. **Keyboard-only navigation** - Complete all critical flows using only keyboard
2. **Screen reader testing** - Test with NVDA (Windows) and VoiceOver (Mac)
3. **High contrast mode** - Test in Windows High Contrast Mode
4. **Zoom testing** - Test at 200% and 400% zoom levels
5. **Color blindness simulation** - Use browser DevTools color vision deficiency simulator

### User Testing

- **Real users** - Include users with disabilities in user testing
- **Assistive tech variety** - Test with multiple screen readers and input devices
- **Accessibility audit** - Annual third-party accessibility audit

## Critical Paths Checklist

All critical trading paths must be fully accessible:

- [ ] **Login/Authentication** - Keyboard navigable, screen reader friendly
- [ ] **View Today's Plan** - All picks readable by screen reader, keyboard navigable
- [ ] **Execute Trade** - Full keyboard flow, clear confirmation dialogs
- [ ] **Adjust Safety (Stop Loss)** - Accessible sliders or numeric input alternatives
- [ ] **View Portfolio** - Sortable/filterable with keyboard, screen reader announces values
- [ ] **Manage Alerts** - Toggle alerts, set quiet hours via keyboard
- [ ] **Update Settings** - All settings accessible, mode toggle clear
- [ ] **View Journal** - Filter/search with keyboard, entries are readable

## Documentation & Training

- **Component library** - Document accessibility features for each component
- **Code examples** - Provide accessible code snippets in documentation
- **Team training** - Quarterly accessibility training for all frontend developers
- **Checklists** - Accessibility checklist in PR template

## Remediation Priority

When issues are found:

1. **P0 (Blocker)** - Prevents critical functionality (fix immediately)
   - Cannot login with keyboard
   - Cannot execute trades with screen reader

2. **P1 (High)** - Significant barrier but workaround exists (fix within sprint)
   - Missing focus indicators
   - Inadequate contrast ratios

3. **P2 (Medium)** - Usability issue but accessible (fix within 2 sprints)
   - Suboptimal tab order
   - Missing ARIA labels on non-critical elements

4. **P3 (Low)** - Enhancement opportunity (backlog)
   - Additional keyboard shortcuts
   - Enhanced screen reader descriptions

## Resources

- [WCAG 2.2 Guidelines](https://www.w3.org/WAI/WCAG22/quickref/)
- [ARIA Authoring Practices Guide](https://www.w3.org/WAI/ARIA/apg/)
- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [a11y Project Checklist](https://www.a11yproject.com/checklist/)
