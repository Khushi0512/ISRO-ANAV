import cv2
import numpy as np
import math
import time

def estimate_mpp(image, image_width=320, fov_deg=60):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var_gray = np.var(gray)

    edges = cv2.Canny(gray, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mpp = 0.06
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        valid_areas = [a for a in areas if a > 10]
        if valid_areas:
            size_px = np.sqrt(np.median(valid_areas))
            mpp = 0.5 / size_px

    if var_gray > 15000:
        mpp *= 0.8
    elif var_gray < 3000:
        mpp *= 1.2

    return max(0.04, min(0.12, mpp))

def generate_heatmap(resized, combined_mask, contours, spots, uniform_thresh, mpp, min_area_m2=0.25):
    """
    Generate a landing zone suitability heatmap based on flatness, brightness, and contour criteria.
    Args:
        resized: Resized input image.
        combined_mask: Combined flatness and brightness mask.
        contours: Detected contours from the combined mask.
        spots: List of valid landing spots.
        uniform_thresh: Threshold for brightness uniformity.
        mpp: Meters-per-pixel value.
        min_area_m2: Minimum area for landing zones in square meters.
    Returns:
        heatmap: Colored heatmap image with overlaid landing zones.
    """
    # Initialize suitability map (grayscale, 0=unsuitable, 255=suitable)
    suitability = np.zeros_like(combined_mask, dtype=np.float32)
    
    # Assign base suitability based on combined mask (flat and bright regions)
    suitability[combined_mask == 255] = 0.5  # Base score for flat and bright regions
    
    # Enhance suitability for valid contours
    min_area_px = min_area_m2 / (mpp ** 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 0.5 / mpp or h < 0.5 / mpp:
            continue
        if not (0.5 < w / h < 2.0):
            continue
        hull = cv2.convexHull(cnt)
        if hull.shape[0] < 3 or (cv2.contourArea(hull) == 0) or (area / cv2.contourArea(hull) < 0.85):
            continue
        mask = np.zeros_like(combined_mask)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        stddev = cv2.meanStdDev(gray, mask=mask)[1][0][0]
        if stddev > uniform_thresh:
            continue
        roi = gray[y:y+h, x:x+w]
        if np.sum(roi < 60) / roi.size > 0.15:
            continue
        # Assign higher suitability to valid contours
        cv2.drawContours(suitability, [cnt], -1, 1.0, -1)
    
    # Normalize suitability to [0, 255]
    suitability = cv2.normalize(suitability, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply color map for visualization
    heatmap = cv2.applyColorMap(suitability, cv2.COLORMAP_JET)
    
    # Overlay detected landing zones
    for spot in spots:
        x, y, w, h = spot['rect']
        color = (0, 0, 255) if spot == min(spots, key=lambda s: s['distance'], default=None) else (0, 255, 0)
        cv2.rectangle(heatmap, (x, y), (x + w, y + h), color, 2)
        cv2.putText(heatmap, f"{w*mpp:.1f}x{h*mpp:.1f}m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save heatmap
    cv2.imwrite("landing_heatmap.jpg", heatmap)
    return heatmap

def detect_drone_landing_spots(frame, min_area_m2=0.25, resize_dims=(320, 240), altitude=5.0):
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame input")

    resized = cv2.resize(frame, resize_dims)
    mpp = estimate_mpp(resized)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l = cv2.normalize(cv2.split(lab)[0], None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = cv2.GaussianBlur(clahe.apply(l), (7, 7), 0)

    gx = cv2.Sobel(l_eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_eq, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    var_grad = np.var(grad)
    grad_thresh = np.percentile(grad, 60 if var_grad < 500 else 55)
    bright_thresh = np.percentile(l_eq, 5)

    flat_mask = (grad < grad_thresh).astype(np.uint8) * 255
    bright_mask = (l_eq > bright_thresh).astype(np.uint8) * 255
    combined_mask = cv2.bitwise_and(flat_mask, bright_mask)

    kernel = np.ones((9, 9), np.uint8)
    cleaned = cv2.morphologyEx(cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = resized.copy()
    spots = []
    detected = 0

    min_area_px = min_area_m2 / (mpp ** 2)
    min_dim = 0.5 / mpp
    uniform_thresh = min(20, np.std(l_eq) * 0.5)
    center = (resize_dims[0] / 2, resize_dims[1] / 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_dim or h < min_dim:
            continue

        if not (0.5 < w / h < 2.0):
            continue

        hull = cv2.convexHull(cnt)
        if hull.shape[0] < 3 or (cv2.contourArea(hull) == 0) or (area / cv2.contourArea(hull) < 0.85):
            continue

        mask = np.zeros_like(l_eq)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        stddev = cv2.meanStdDev(l_eq, mask=mask)[1][0][0]
        if stddev > uniform_thresh:
            continue

        roi = l_eq[y:y+h, x:x+w]
        if np.sum(roi < 60) / roi.size > 0.15:
            continue

        cx, cy = x + w / 2, y + h / 2
        dist = math.hypot(cx - center[0], cy - center[1])

        spots.append({
            'center': (cx, cy),
            'distance': dist,
            'rect': (x, y, w, h),
            'size_m': (w * mpp, h * mpp)
        })

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, f"{w*mpp:.1f}x{h*mpp:.1f}m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        detected += 1

    nearest = min(spots, key=lambda s: s['distance'], default=None)
    if nearest:
        x, y, w, h = nearest['rect']
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output, "Nearest", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Generate heatmap
    heatmap = generate_heatmap(resized, combined_mask, contours, spots, uniform_thresh, mpp, min_area_m2)

    return output, nearest, mpp, heatmap

def send_mavlink_command(spot, mpp, altitude=5.0, image=None, resize_dims=(320, 240)):
    if spot is None:
        print("No valid landing spot.")
        return

    cx, cy = resize_dims[0] / 2, resize_dims[1] / 2
    tx, ty = spot['center']  # Fixed syntax error
    dx, dy = tx - cx, ty - cy

    dx_m, dy_m = dx * mpp, -dy * mpp
    dz = -altitude
    speed = 0.3
    descent = 0.01
    steps = 50
    dt = 0.1

    print(f"Flying to ({dx_m:.2f}, {dy_m:.2f}, {dz:.2f})")

    vis = image.copy() if image is not None else np.zeros((resize_dims[1], resize_dims[0], 3), dtype=np.uint8)

    for i in range(steps + 1):
        cx += dx / steps
        cy += dy / steps
        altitude += dz / steps

        vis_frame = vis.copy()
        cv2.circle(vis_frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        x, y, w, h = spot['rect']
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(vis_frame, f"Alt: {altitude:.1f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Drone Landing Simulation", vis_frame)
        cv2.waitKey(int(dt * 1000))

    print("Landing...")
    time.sleep(2)
    print("Landed successfully.")

    cv2.putText(vis_frame, "Landed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Drone Landing Simulation", vis_frame)
    cv2.waitKey(1000)
    cv2.imwrite("landing_simulation_final.jpg", vis_frame)

if __name__ == "__main__":
    image_path = r"C:\MEET\B.Tech_Projects\IROC_ISRO\Codes\test_pic_2.jpg"  # Updated path
    img = cv2.imread(image_path)

    if img is None:
        print("Image not found.")
        exit()

    try:
        result, nearest, mpp, heatmap = detect_drone_landing_spots(img)
        cv2.imwrite("landing_zones_output.jpg", result)
        cv2.imshow("Drone Landing Zones", result)
        cv2.imshow("Landing Zone Suitability Heatmap", heatmap)

        sim_img = cv2.resize(img, (320, 240))
        send_mavlink_command(nearest, mpp, image=sim_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()