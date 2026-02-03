# Admin Mode Feature - GUI Simplification

## Overview
Added an "Admin Mode" toggle button to simplify the GUI by hiding advanced configuration options while keeping essential features visible.

## Changes Made

### 1. Admin Mode Button
- **Location:** Top-right corner of the header
- **Default State:** Admin mode is OFF (simplified view)
- **Functionality:** Toggles between simplified and full admin view

### 2. Hidden Elements (Admin Mode OFF)
When admin mode is disabled, the following sections are hidden:

#### SAM Configuration (Partially Hidden)
- ✅ **Visible:** "Show Dark Edge Preview" checkbox and controls
- ✅ **Visible:** "Run SAM Segmentation" button
- ❌ **Hidden:** Model Size selector
- ❌ **Hidden:** Backend selection
- ❌ **Hidden:** Crop Layers slider
- ❌ **Hidden:** Points per Side slider
- ❌ **Hidden:** Overlap Filter controls
- ❌ **Hidden:** Circularity Filter checkbox and controls
- ❌ **Hidden:** Advanced Options toggle

#### Completely Hidden Panels
- ❌ Pre-Segmentation Filter panel
- ❌ Resolution Enhancement (ESRGAN) panel
- ❌ Intensity Filter panel
- ❌ Circularity Filter panel
- ❌ Dark Edge Filter panel

### 3. Admin Mode ON
When admin mode is enabled:
- All hidden sections become available
- All configuration options are visible
- JavaScript controls work normally to show/hide panels as needed

## User Experience

### Simplified View (Default)
Users see only:
1. Image upload section
2. "Show Dark Edge Preview" option
3. "Run SAM Segmentation" button
4. Results and export options (after segmentation)

### Admin View (When Enabled)
Users see all features:
- Full SAM configuration options
- All filter panels
- All advanced settings

## Technical Implementation

### CSS Classes
- `.admin-mode-only` - Applied to elements that should be hidden in simplified mode
- `.admin-mode-active` - Applied to container when admin mode is ON

### JavaScript Function
```javascript
toggleAdminMode() - Toggles admin mode on/off
```

### Button States
- **OFF:** "🔧 Admin Mode" (white/transparent)
- **ON:** "👁️ Hide Admin" (yellow highlight)

## Benefits

1. **Simplified Interface:** New users see only essential features
2. **Reduced Confusion:** Less overwhelming for basic use cases
3. **Full Control:** Advanced users can enable admin mode for all options
4. **Better UX:** Cleaner, more focused interface by default

## Usage

1. Click "Admin Mode" button in top-right corner to enable
2. All hidden panels and options become available
3. Click "Hide Admin" to return to simplified view

## Files Modified

- `templates/sam_website.html`
  - Added Admin Mode button in header
  - Added CSS for admin mode visibility control
  - Added JavaScript toggle function
  - Wrapped advanced sections with `admin-mode-only` class
  - Kept essential features (Dark Edge Preview, Run SAM) always visible
