import cv2
import numpy as np

def detect_drone_landing_spots(frame, min_area_m2=0.25, resize_dims=(320, 240), meters_per_pixel=0.1):
    """
    Detect safe drone landing spots in an image, optimized for challenging terrains like Martian surfaces.
    
    Args:
        frame: Input BGR image
        min_area_m2: Minimum landing area in square meters (default: 0.25 m^2)
        resize_dims: Tuple of (width, height) for resized image
        meters_per_pixel: Calibration factor for converting pixels to meters
    
    Returns:
        Combined debug image with detected landing spots and intermediate processing steps
    """
    # Resize image
    resized = cv2.resize(frame, resize_dims)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)

    # Contrast stretching to enhance brightness differences
    l_channel = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)

    # Enhance brightness with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)

    # Apply Gaussian blur to reduce noise
    l_eq = cv2.GaussianBlur(l_eq, (5, 5), 0)

    # Compute flatness using Sobel gradients
    grad_x = cv2.Sobel(l_eq, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l_eq, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Percentile-based thresholding for flatness and brightness
    grad_thresh = np.percentile(grad_norm, 15)  # Consider bottom 15% as flat
    bright_thresh = np.percentile(l_eq, 30)  # Consider top 50% as bright
    
    flat_mask = (grad_norm < grad_thresh).astype(np.uint8) * 255
    bright_mask = (l_eq > bright_thresh).astype(np.uint8) * 255
    combined_mask = cv2.bitwise_and(flat_mask, bright_mask)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = resized.copy()

    detected = 0
    min_area_pixels = min_area_m2 / (meters_per_pixel ** 2)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_pixels:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h
        margin = 5

        # Further relaxed aspect ratio and size checks, removed circularity check
        if (0.4 < aspect < 2.5 and w > 5 and h > 5 and
            x > margin and y > margin and x + w < resize_dims[0] - margin and y + h < resize_dims[1] - margin):
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            width_m = round(w * meters_per_pixel, 1)
            height_m = round(h * meters_per_pixel, 1)
            cv2.putText(output, f"{width_m}x{height_m}m", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            detected += 1

    print(f"Safe landing spots detected: {detected}")

    # Create debug visualization
    grad_colormap = cv2.applyColorMap(grad_norm, cv2.COLORMAP_JET)
    flat_bgr = cv2.cvtColor(flat_mask, cv2.COLOR_GRAY2BGR)
    bright_bgr = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR)
    mask_combined_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    # Add labels to debug images
    cv2.putText(output, "Detected Spots", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(mask_combined_bgr, "Combined Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(grad_colormap, "Gradient Map", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(flat_bgr, "Flatness Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(bright_bgr, "Brightness Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(resized, "Original Image", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    top_row = np.hstack((output, mask_combined_bgr, grad_colormap))
    bottom_row = np.hstack((flat_bgr, bright_bgr, cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
    combined = np.vstack((top_row, bottom_row))

    return combined

if __name__ == "__main__":
    image_path = r"C:\MEET\IROC_ISRO\Codes\test_pic_3.jpg"
    frame = cv2.imread(image_path)

    if frame is None:
        print("Unable to load image")
        exit()

    # Adjusted parameters for Martian terrain
    result = detect_drone_landing_spots(frame, min_area_m2=0.25, resize_dims=(320, 240), meters_per_pixel=0.1)

    cv2.imshow("Drone Safe Spot Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
