import cv2
import numpy as np

def detect_landing_spots(frame):
    # Resize and grayscale
    resized = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Invert edges and blur to find smooth areas
    inverted = cv2.bitwise_not(edges)
    blurred = cv2.GaussianBlur(inverted, (15, 15), 0)

    # Threshold to find flat zones
    _, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Morphological clean-up
    kernel = np.ones((5, 5), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = resized.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue  # too small to land

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        # Landing spot should not be too elongated
        if 0.75 < aspect_ratio < 1.5 and w > 40 and h > 40:
            # Optional: check edge density inside region
            region_edges = edges[y:y+h, x:x+w]
            edge_density = np.sum(region_edges > 0) / (w * h)
            if edge_density < 0.02:  # Less than 2% edges
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output, "Landing Spot", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return output, thresholded, edges


# Load image
frame = cv2.imread(r"C:\MEET\VS CODE\ISRO_ANAV\test_pic_0.jpg")

if frame is None:
    print("❌ Image not found. Check the path and filename.")
    exit()

result, flat_mask, edges = detect_landing_spots(frame)

flat_mask_colored = cv2.cvtColor(flat_mask, cv2.COLOR_GRAY2BGR)
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

combined = np.hstack((result, flat_mask_colored, edges_colored))

cv2.imshow("Drone Landing Spots | Flat Areas | Edges", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
