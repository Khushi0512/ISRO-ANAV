import cv2
import numpy as np
import math
import time

def estimate_mpp(image, image_width=320, fov_deg=60):
    """Rough estimate of meters-per-pixel based on texture and edge details."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)

    edges = cv2.Canny(gray, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        median_area = np.median([a for a in areas if a > 10])
        if median_area > 0:
            size_px = np.sqrt(median_area)
            mpp = 0.5 / size_px
        else:
            mpp = 0.06
    else:
        mpp = 0.06

    if variance > 15000:
        mpp *= 0.8
    elif variance < 3000:
        mpp *= 1.2

    return max(0.04, min(0.12, mpp))

def detect_drone_landing_spots(frame, min_area_m2=0.25, resize_dims=(320, 240), altitude=5.0):
    """Detects potential drone landing zones, highlights all in green, nearest in red, and returns the nearest one."""
    if frame is None or frame.size == 0:
        raise ValueError("Invalid input frame")

    resized = cv2.resize(frame, resize_dims)
    meters_per_pixel = estimate_mpp(resized)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel = cv2.split(lab)[0]
    l_channel = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    l_eq = cv2.GaussianBlur(l_eq, (7, 7), 0)

    grad_x = cv2.Sobel(l_eq, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l_eq, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    grad_var = np.var(grad_norm)
    grad_thresh = np.percentile(grad_norm, 60 if grad_var < 500 else 55)
    bright_thresh = np.percentile(l_eq, 5)

    flat_mask = (grad_norm < grad_thresh).astype(np.uint8) * 255
    bright_mask = (l_eq > bright_thresh).astype(np.uint8) * 255
    combined_mask = cv2.bitwise_and(flat_mask, bright_mask)

    kernel = np.ones((9, 9), np.uint8)
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = resized.copy()
    detected = 0
    landing_spots = []

    min_area_px = min_area_m2 / (meters_per_pixel ** 2)
    min_size_px = 0.5 / meters_per_pixel
    uniformity_thresh = min(20, np.std(l_eq) * 0.5)
    image_center = (resize_dims[0] / 2, resize_dims[1] / 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_size_px or h < min_size_px:
            continue

        aspect = w / h
        if not (0.5 < aspect < 2.0):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0 or (area / hull_area) < 0.85:
            continue

        mask = np.zeros_like(l_eq)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        stddev = cv2.meanStdDev(l_eq, mask=mask)[1][0][0]
        if stddev > uniformity_thresh:
            continue

        region = l_eq[y:y+h, x:x+w]
        if np.sum(region < 60) / region.size > 0.15:
            continue

        # Calculate center and distance
        center_x = x + w / 2
        center_y = y + h / 2
        distance = math.sqrt((center_x - image_center[0])**2 + (center_y - image_center[1])**2)
        landing_spots.append({
            'center': (center_x, center_y),
            'distance': distance,
            'rect': (x, y, w, h),
            'size_m': (w * meters_per_pixel, h * meters_per_pixel)
        })

        # Highlight all safe spots in green
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
        cv2.putText(output, f"{w*meters_per_pixel:.1f}x{h*meters_per_pixel:.1f}m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        detected += 1

    # Fallback for rough terrain
    if detected < 3 and grad_var > 500:
        grad_thresh = np.percentile(grad_norm, 50)
        flat_mask = (grad_norm < grad_thresh).astype(np.uint8) * 255
        combined_mask = cv2.bitwise_and(flat_mask, bright_mask)
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < min_size_px or h < min_size_px:
                continue

            aspect = w / h
            if not (0.5 < aspect < 2.0):
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0 or (area / hull_area) < 0.8:
                continue

            mask = np.zeros_like(l_eq)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            stddev = cv2.meanStdDev(l_eq, mask=mask)[1][0][0]
            if stddev > uniformity_thresh * 1.2:
                continue

            region = l_eq[y:y+h, x:x+w]
            if np.sum(region < 60) / region.size > 0.2:
                continue

            center_x = x + w / 2
            center_y = y + h / 2
            distance = math.sqrt((center_x - image_center[0])**2 + (center_y - image_center[1])**2)
            landing_spots.append({
                'center': (center_x, center_y),
                'distance': distance,
                'rect': (x, y, w, h),
                'size_m': (w * meters_per_pixel, h * meters_per_pixel)
            })

            # Highlight all safe spots in green
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
            cv2.putText(output, f"{w*meters_per_pixel:.1f}x{h*meters_per_pixel:.1f}m", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            detected += 1

    # Highlight the nearest landing spot in red
    nearest_spot = None
    if landing_spots:
        nearest_spot = min(landing_spots, key=lambda x: x['distance'])
        x, y, w, h = nearest_spot['rect']
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red rectangle, thicker for emphasis
        cv2.putText(output, "Nearest", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    print(f"Safe landing spots detected: {detected}")
    print(f"Estimated meters per pixel: {meters_per_pixel:.3f}")
    if nearest_spot:
        print(f"Nearest spot at ({nearest_spot['center'][0]:.1f}, {nearest_spot['center'][1]:.1f}) px, "
              f"distance: {nearest_spot['distance']*meters_per_pixel:.2f} m, "
              f"size: {nearest_spot['size_m'][0]:.1f}x{nearest_spot['size_m'][1]:.1f} m")

    grad_colormap = cv2.applyColorMap(grad_norm, cv2.COLORMAP_JET)
    flat_bgr = cv2.cvtColor(flat_mask, cv2.COLOR_GRAY2BGR)
    bright_bgr = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR)
    mask_combined_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    cv2.putText(output, "Detected Spots", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(mask_combined_bgr, "Combined Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(grad_colormap, "Gradient Map", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(flat_bgr, "Flatness Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bright_bgr, "Brightness Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(resized, "Original Image", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    top_row = np.hstack((output, mask_combined_bgr, grad_colormap))
    bottom_row = np.hstack((flat_bgr, bright_bgr, cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
    combined = np.vstack((top_row, bottom_row))

    return combined, nearest_spot, meters_per_pixel

def send_mavlink_command(spot, meters_per_pixel, altitude=5.0):
    """Simulate MAVLink commands for navigation and landing (laptop testing)."""
    if spot is None:
        print("No landing spot detected, cannot generate command.")
        return

    # Calculate displacement in meters (NED frame)
    center_x, center_y = spot['center']
    image_center = (320 / 2, 240 / 2)  # Assuming resize_dims=(320, 240)
    dx_px = center_x - image_center[0]
    dy_px = center_y - image_center[1]
    dx_m = dx_px * meters_per_pixel  # North (positive = forward)
    dy_m = -dy_px * meters_per_pixel  # East (positive = right, invert for camera orientation)

    # Simulate position command
    print(f"Simulated command to move to ({dx_m:.2f}, {dy_m:.2f}, {-altitude:.2f}) m")

    # Simulate delay for reaching target
    print("Simulating 5-second wait for drone to reach target...")
    time.sleep(5)

    # Simulate landing command
    print("Simulated command to land at the nearest spot")

if __name__ == "__main__":
    # Load and process image
    image_path = r"C:\MEET\IROC_ISRO\Codes\test_pic_2.jpg"
    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Unable to load image. Please check the file path.")
        exit()

    try:
        result, nearest_spot, mpp = detect_drone_landing_spots(frame, altitude=5.0)
        cv2.imwrite("landing_zones_output.jpg", result)  # Save output for inspection
        cv2.imshow("Drone Landing Zones", result)
        send_mavlink_command(nearest_spot, mpp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        cv2.destroyAllWindows()
