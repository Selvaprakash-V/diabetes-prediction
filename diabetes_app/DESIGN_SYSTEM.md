# Premium Apple-Inspired Design System
## Diabetes Risk Assessment - Luxury Health Dashboard

---

## 🎨 Design Philosophy

This application follows Apple's minimalist design principles to create a premium, calming health assessment experience. Every element is intentionally crafted to convey trust, clarity, and sophistication.

### Core Principles

1. **Minimal & Clean** - Remove unnecessary elements, focus on essential information
2. **Large Typography** - Hero titles and clear hierarchy guide the eye
3. **Smooth Animations** - Fade-in effects create fluid, intentional transitions
4. **Ample Whitespace** - Breathing room between elements reduces cognitive load
5. **Soft Shadows** - Subtle depth without overwhelming the interface
6. **Premium Colors** - Calming blues, whites, and accent colors inspire confidence

---

## 🎨 Color Palette

### Primary Colors
- **Primary Blue**: `#007AFF` - Primary actions, links, and emphasis
- **Primary Dark**: `#0051D5` - Button gradients and hover states
- **Accent Teal**: `#00C7BE` - Secondary accent for variety

### Semantic Colors
- **Success Green**: `#34C759` - Positive outcomes, healthy status
- **Warning Orange**: `#FF9500` - Elevated risk, caution messages
- **Error Red**: `#FF3B30` - Critical items, foods to avoid

### Neutrals
- **Background**: `#FAFAFA` - Main background with subtle gradient
- **Card White**: `#FFFFFF` - Pure white for cards and containers
- **Text Primary**: `#1D1D1F` - Main text, high contrast
- **Text Secondary**: `#86868B` - Supporting text, descriptions
- **Text Tertiary**: `#B0B0B5` - Disabled states, footer text

---

## 📐 Spacing System

Consistent spacing creates rhythm and balance:

- **XS**: `8px` - Tight spacing within components
- **SM**: `16px` - Standard padding for small cards
- **MD**: `24px` - Medium spacing between sections
- **LG**: `48px` - Large gaps between major sections
- **XL**: `72px` - Hero section spacing
- **XXL**: `120px` - Maximum spacing for drama

---

## 🔤 Typography

### Font Family
```css
-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif
```

### Type Scale

#### Hero Title
- **Size**: `clamp(2.5rem, 5vw, 4rem)` (40-64px responsive)
- **Weight**: 700 (Bold)
- **Letter Spacing**: -0.03em (tighter for large text)
- **Usage**: Main page titles

#### Section Title
- **Size**: `clamp(1.75rem, 3vw, 2.5rem)` (28-40px responsive)
- **Weight**: 600 (Semi-bold)
- **Letter Spacing**: -0.02em
- **Usage**: Section headers

#### Subtitle
- **Size**: `clamp(1.125rem, 2vw, 1.5rem)` (18-24px responsive)
- **Weight**: 400 (Regular)
- **Color**: Secondary text color
- **Usage**: Hero descriptions

#### Body Text
- **Size**: `1.0625rem` (17px)
- **Weight**: 400
- **Line Height**: 1.7
- **Usage**: Main content, descriptions

---

## 🎴 Components

### Premium Card
```css
- Background: White with subtle gradient
- Padding: 48px (large) for premium feel
- Border Radius: 24px (generous curves)
- Shadow: 0 16px 48px rgba(0, 0, 0, 0.08)
- Border: 1px solid rgba(255, 255, 255, 0.8)
- Animation: Fade in + scale up
- Hover: Lift up 8px with deeper shadow
```

**Usage**: Main content containers, result cards

### Glass Card
```css
- Background: Semi-transparent white
- Backdrop Filter: Blur(40px) for glass effect
- Usage: Overlays, floating elements
```

### Feature Card
```css
- Background: Linear gradient white → light gray
- Padding: 24px
- Border Radius: 18px
- Shadow: Subtle 0 2px 8px
- Hover: Lift up 4px
```

**Usage**: Food recommendations, smaller content blocks

---

## 🔘 Buttons

### Primary Button
```css
- Background: Linear gradient (Primary → Primary Dark)
- Color: White
- Border Radius: 18px
- Padding: 16px 48px
- Height: 56px (touch-friendly)
- Font Size: 17px
- Font Weight: 600
- Shadow: 0 4px 16px rgba(0, 122, 255, 0.3)
- Hover: Lift up 2px, deeper shadow
- Transition: 0.3s cubic-bezier (smooth easing)
```

### Secondary Button
```css
- Background: Light gray
- Color: Primary text
- All other properties same as primary
```

---

## 📊 Data Visualization

### Charts (Plotly)
- **Background**: Transparent to blend with cards
- **Colors**: Primary blue and warning orange
- **Font**: System font stack
- **Grid Lines**: Subtle rgba(0,0,0,0.05)
- **Remove**: Toolbars (displayModeBar: false)

### Donut Chart
- **Hole Size**: 0.7 (large center)
- **Border**: White 3px between segments
- **Center Text**: Large confidence percentage
- **Legend**: Horizontal, centered below

---

## ✨ Animations

### Fade In Up
```css
@keyframes fadeInUp {
    from: opacity 0, translateY(30px)
    to: opacity 1, translateY(0)
    duration: 1s
    easing: ease-out
}
```

**Usage**: Hero titles, subtitles

### Fade In Scale
```css
@keyframes fadeInScale {
    from: opacity 0, scale(0.95)
    to: opacity 1, scale(1)
    duration: 0.8s
    easing: ease-out
}
```

**Usage**: Cards, main content

### Staggered Delays
- Element 1: No delay
- Element 2: 0.2s delay
- Element 3: 0.4s delay
- Element 4: 0.6s delay

Creates cascading effect as content loads.

---

## 🎯 Form Elements

### Input Fields
```css
- Background: White
- Border: 2px solid rgba(0,0,0,0.06)
- Border Radius: 12px
- Padding: 16px
- Focus: Blue border + 4px blue glow (rgba)
- Transition: All 0.3s ease
```

### Sliders
- Track: Light blue background
- Fill: Primary blue gradient
- Thumb: Large, easy to grab

---

## 📱 Responsive Design

### Breakpoints
- **Mobile**: < 768px
  - Reduce font sizes via clamp()
  - Decrease card padding
  - Stack columns vertically
  - Smaller hero spacing

### Touch Targets
- Minimum: 44x44px (iOS guidelines)
- Buttons: 56px height for comfort

---

## 🎭 Semantic Alerts

### Success
```css
- Background: Linear gradient green with 10% → 5% opacity
- Border Left: 4px solid success green
- Icon: ✓
```

### Warning
```css
- Background: Linear gradient orange with 10% → 5% opacity
- Border Left: 4px solid warning orange
- Icon: !
```

### Error
```css
- Background: Linear gradient red with 10% → 5% opacity
- Border Left: 4px solid error red
- Icon: ⚠
```

### Info
```css
- Background: Linear gradient blue with 10% → 5% opacity
- Border Left: 4px solid primary blue
- Icon: ℹ
```

---

## 🏗️ Layout Patterns

### Hero Container
```python
st.columns([1, 2, 1])  # Centered content
```
- Text centered
- Max-width constrained
- Large vertical padding

### Wide Container
- Max-width: 1400px
- Horizontal centering
- For dashboard layouts

### Centered Container
- Max-width: 800px
- For forms and focused content

---

## 🎨 Icon System

### Icon Wrapper
```css
- Size: 64x64px
- Background: Gradient (Primary → Accent)
- Border Radius: 50% (perfect circle)
- Shadow: Medium depth
- Centered alignment
- Font Size: 2rem for emoji icons
```

**Usage**: Status indicators, feature highlights

---

## 🖼️ Page Structure

### 1. Home Page
- Hero title + subtitle
- Centered form in premium card
- Two-column input layout
- Progress indicator
- Dataset insights expander
- Footer

### 2. Prediction Result
- Hero title + subtitle
- Status card with icon
- Two-column analysis:
  - Left: Probability donut chart
  - Right: Key metrics cards
- Population comparison scatter plot
- Action buttons
- Disclaimer footer

### 3. Diet Recommendations
- Hero title + subtitle
- Status overview
- Two-column layout:
  - Left: Food recommendations (2/3 width)
  - Right: Tips + Foods to limit (1/3 width)
- Weekly goals selector
- Navigation buttons
- Footer

---

## 🚀 Performance

### Optimizations
- CSS loaded once, cached
- Animations GPU-accelerated (transform/opacity)
- Plotly displayModeBar disabled
- Lazy loading via Streamlit's native behavior

---

## ♿ Accessibility

### Best Practices
- High contrast text (WCAG AA+)
- Touch targets ≥ 44px
- Focus states visible
- Semantic HTML structure
- Alt text for icons (via emoji)
- Keyboard navigable

---

## 📦 File Structure

```
diabetes_app/
├── assets/
│   └── premium_style.css      # All custom styles
├── Home.py                     # Landing page
└── pages/
    ├── 1_Prediction_Result.py # Analysis page
    └── 2_Diet_Recommendations.py # Diet page
```

---

## 🎯 Key Takeaways

1. **Consistency**: Every spacing, color, and animation follows the system
2. **Hierarchy**: Large → Small text guides attention naturally
3. **Whitespace**: Space is a design element, not empty area
4. **Animations**: Smooth, intentional, never jarring
5. **Premium Feel**: Every detail considered, nothing accidental
6. **User Trust**: Clean design = professional = trustworthy health tool

---

## 🛠️ Implementation Notes

### Loading CSS
```python
CSS_PATH = BASE_DIR / 'diabetes_app' / 'assets' / 'premium_style.css'
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
```

### Using Classes
```python
st.markdown('<div class="premium-card fade-in">...</div>', unsafe_allow_html=True)
```

### Staggered Animations
```python
fade-in          # No delay
fade-in-delay-1  # 0.2s
fade-in-delay-2  # 0.4s
fade-in-delay-3  # 0.6s
```

---

## 🎨 Inspiration Sources

- [Apple.com](https://www.apple.com) - Product pages, hero sections
- [iPlant Concept](https://www.behance.net/search/projects?search=plant%20concept) - Health app aesthetics
- iOS Human Interface Guidelines - Typography, spacing, colors
- Material Design 3 - Glass morphism effects

---

**Designed with ❤️ for a premium health experience**
