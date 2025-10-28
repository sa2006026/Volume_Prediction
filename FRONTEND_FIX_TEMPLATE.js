/**
 * FRONTEND FIX FOR BOUNDING BOX UPDATE ISSUE
 * 
 * This file contains the correct JavaScript pattern to ensure bounding boxes
 * are updated when intensity filter is applied.
 * 
 * PROBLEM: After applying intensity filter, bounding boxes don't update
 * SOLUTION: Always update currentMasks from backend response
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

// Single source of truth for current masks
let currentMasks = [];

// Canvas references (adjust IDs to match your HTML)
let imageCanvas, boxCanvas, imageCtx, boxCtx;

// Initialize canvases
function initializeCanvases() {
    imageCanvas = document.getElementById('imageCanvas'); // Your image canvas ID
    boxCanvas = document.getElementById('boxCanvas');     // Your bounding box canvas ID
    
    if (!imageCanvas || !boxCanvas) {
        console.error('❌ Canvas elements not found! Update canvas IDs in code.');
        return false;
    }
    
    imageCtx = imageCanvas.getContext('2d');
    boxCtx = boxCanvas.getContext('2d');
    return true;
}

// ============================================================================
// CORE FUNCTIONS
// ============================================================================

/**
 * Clear all bounding boxes from the canvas
 */
function clearAllBoundingBoxes() {
    if (!boxCtx) return;
    boxCtx.clearRect(0, 0, boxCanvas.width, boxCanvas.height);
    console.log('🧹 Cleared all bounding boxes');
}

/**
 * Draw a single bounding box
 */
function drawBoundingBox(x, y, w, h, maskId, color = '#00ff00') {
    if (!boxCtx) return;
    
    boxCtx.strokeStyle = color;
    boxCtx.lineWidth = 2;
    boxCtx.strokeRect(x, y, w, h);
    
    // Optional: Draw mask ID label
    boxCtx.fillStyle = color;
    boxCtx.font = '12px Arial';
    boxCtx.fillText(`#${maskId}`, x + 2, y + 12);
}

/**
 * Redraw all bounding boxes from currentMasks array
 * THIS IS THE KEY FUNCTION - it always draws from currentMasks
 */
function redrawBoundingBoxes() {
    console.log(`🎨 Redrawing bounding boxes from currentMasks (${currentMasks.length} masks)`);
    
    // Step 1: Clear ALL existing boxes
    clearAllBoundingBoxes();
    
    // Step 2: Draw ONLY the masks in currentMasks
    let drawnCount = 0;
    currentMasks.forEach(mask => {
        if (mask.bounding_box && Array.isArray(mask.bounding_box) && mask.bounding_box.length >= 4) {
            const [x, y, w, h] = mask.bounding_box;
            drawBoundingBox(x, y, w, h, mask.mask_id);
            drawnCount++;
        }
    });
    
    console.log(`✅ Drew ${drawnCount} bounding boxes`);
}

// ============================================================================
// API FUNCTIONS
// ============================================================================

/**
 * 1. Upload Image
 */
async function uploadImage(file) {
    console.log('📤 Uploading image...');
    
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch('/upload_image', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // ✅ Update currentMasks (should be empty array)
            currentMasks = data.masks || [];
            console.log(`✅ Image uploaded. Masks: ${currentMasks.length}`);
            
            // Load and display image
            await loadImage(data.image);
            
            // Redraw (will clear all boxes since currentMasks is empty)
            redrawBoundingBoxes();
            
            showMessage(data.message);
        } else {
            console.error('❌ Upload failed:', data.error);
            alert('Upload failed: ' + data.error);
        }
    } catch (error) {
        console.error('❌ Upload error:', error);
        alert('Upload error: ' + error.message);
    }
}

/**
 * 2. Run Segmentation
 */
async function runSegmentation() {
    console.log('🔬 Running segmentation...');
    
    const params = {
        model_size: 'vit_b',
        crop_layers: 1,
        points_per_side: 32,
        apply_overlap_filter: true,
        overlap_threshold: 0.4
    };
    
    try {
        const response = await fetch('/run_sam_segmentation', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(params)
        });
        
        const data = await response.json();
        
        console.log('📥 Segmentation response:', data);
        
        if (data.success && data.masks_found) {
            // ✅ CRITICAL: Update currentMasks with segmentation results
            currentMasks = data.masks || [];
            console.log(`✅ Segmentation complete. Masks: ${currentMasks.length}`);
            
            // Load image
            await loadImage(data.overlay_image);
            
            // Redraw with all segmented masks
            redrawBoundingBoxes();
            
            showMessage(data.message);
        } else {
            console.warn('⚠️ No masks found');
            currentMasks = [];
            redrawBoundingBoxes();
        }
    } catch (error) {
        console.error('❌ Segmentation error:', error);
        alert('Segmentation error: ' + error.message);
    }
}

/**
 * 3. Apply Intensity Filter
 * THIS IS THE KEY FUNCTION THAT FIXES YOUR ISSUE
 */
async function applyIntensityFilter(minIntensity, maxIntensity) {
    console.log(`🔍 Applying intensity filter: ${minIntensity}-${maxIntensity}`);
    console.log(`🔍 Current masks BEFORE filter: ${currentMasks.length}`);
    
    try {
        const response = await fetch('/apply_intensity_filter', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                min_intensity: parseInt(minIntensity),
                max_intensity: parseInt(maxIntensity)
            })
        });
        
        const data = await response.json();
        
        console.log('📥 Intensity filter response:', data);
        console.log(`📥 Masks in response: ${data.masks ? data.masks.length : 'MISSING!'}`);
        
        if (data.success) {
            // ✅ CRITICAL: Update currentMasks with filtered results
            // This is the line that was probably missing in your code!
            currentMasks = data.masks || [];
            
            console.log(`✅ Current masks AFTER filter: ${currentMasks.length}`);
            console.log(`📊 Total: ${data.total_masks}, Kept: ${data.masks_count}, Filtered: ${data.filtered_count}`);
            
            // Load image
            await loadImage(data.image);
            
            // Redraw - now only draws the filtered masks
            redrawBoundingBoxes();
            
            showMessage(data.message);
        } else {
            console.error('❌ Filter failed:', data.error);
            alert('Filter failed: ' + data.error);
        }
    } catch (error) {
        console.error('❌ Filter error:', error);
        alert('Filter error: ' + error.message);
    }
}

/**
 * 4. Reset Intensity Filter
 */
async function resetIntensityFilter() {
    console.log('🔄 Resetting intensity filter...');
    console.log(`🔄 Current masks BEFORE reset: ${currentMasks.length}`);
    
    try {
        const response = await fetch('/reset_intensity_filter', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        });
        
        const data = await response.json();
        
        console.log('📥 Reset filter response:', data);
        
        if (data.success) {
            // ✅ Update currentMasks with all masks restored
            currentMasks = data.masks || [];
            
            console.log(`✅ Current masks AFTER reset: ${currentMasks.length}`);
            
            // Load image
            await loadImage(data.image);
            
            // Redraw - now draws all masks again
            redrawBoundingBoxes();
            
            showMessage(data.message);
        } else {
            console.error('❌ Reset failed:', data.error);
            alert('Reset failed: ' + data.error);
        }
    } catch (error) {
        console.error('❌ Reset error:', error);
        alert('Reset error: ' + error.message);
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Load and display an image
 */
async function loadImage(base64Image) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            // Resize canvases to match image
            imageCanvas.width = img.width;
            imageCanvas.height = img.height;
            boxCanvas.width = img.width;
            boxCanvas.height = img.height;
            
            // Draw image
            imageCtx.drawImage(img, 0, 0);
            
            console.log(`✅ Image loaded: ${img.width}x${img.height}`);
            resolve();
        };
        img.onerror = reject;
        img.src = base64Image;
    });
}

/**
 * Show a message to the user
 */
function showMessage(message) {
    console.log('💬 ' + message);
    // Update your UI message display here
    // Example:
    // document.getElementById('messageDiv').textContent = message;
}

// ============================================================================
// EVENT HANDLERS - Wire these up to your UI elements
// ============================================================================

function setupEventHandlers() {
    // Image upload
    const uploadInput = document.getElementById('imageUpload');
    if (uploadInput) {
        uploadInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) uploadImage(file);
        });
    }
    
    // Run segmentation button
    const segmentBtn = document.getElementById('runSegmentation');
    if (segmentBtn) {
        segmentBtn.addEventListener('click', () => runSegmentation());
    }
    
    // Apply intensity filter button
    const applyFilterBtn = document.getElementById('applyFilter');
    if (applyFilterBtn) {
        applyFilterBtn.addEventListener('click', () => {
            const minIntensity = document.getElementById('minIntensity').value;
            const maxIntensity = document.getElementById('maxIntensity').value;
            applyIntensityFilter(minIntensity, maxIntensity);
        });
    }
    
    // Reset filter button
    const resetFilterBtn = document.getElementById('resetFilter');
    if (resetFilterBtn) {
        resetFilterBtn.addEventListener('click', () => resetIntensityFilter());
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing SAM Web Interface...');
    
    if (initializeCanvases()) {
        setupEventHandlers();
        console.log('✅ SAM Web Interface initialized');
    } else {
        console.error('❌ Failed to initialize canvases');
    }
});

// ============================================================================
// DEBUGGING FUNCTIONS
// ============================================================================

/**
 * Debug function to check current state
 * Call this in browser console: debugMaskState()
 */
function debugMaskState() {
    console.log('=== DEBUG: Current State ===');
    console.log('Current masks count:', currentMasks.length);
    console.log('First 3 masks:', currentMasks.slice(0, 3));
    console.log('Canvas sizes:', {
        image: {width: imageCanvas?.width, height: imageCanvas?.height},
        box: {width: boxCanvas?.width, height: boxCanvas?.height}
    });
    
    // Check backend state
    fetch('/debug_mask_states')
        .then(r => r.json())
        .then(data => {
            console.log('=== Backend State ===');
            console.log('Total masks:', data.total_masks);
            console.log('State counts:', data.state_counts);
            console.log('First 3 mask details:', data.mask_details?.slice(0, 3));
        })
        .catch(e => console.error('Failed to fetch backend state:', e));
}

// Make debug function available globally
window.debugMaskState = debugMaskState;
window.currentMasks = currentMasks;  // For inspection in console

console.log('💡 TIP: Type debugMaskState() in console to check current state');

