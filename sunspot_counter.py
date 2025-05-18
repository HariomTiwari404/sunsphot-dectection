import os
from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.cluster import DBSCAN
import math

def preprocess_image(image):
    """
    Apply preprocessing to enhance sunspot visibility
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # Use adaptive thresholding to highlight features
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Perform morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Return both the preprocessed binary image and the equalized version
    return opening, equalized

def estimate_solar_radius(image):
    """
    Estimate the radius of the solar disk in the image
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply threshold to separate disk from background
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (should be the solar disk)
    if not contours:
        # If no contours found, estimate based on image size
        height, width = gray.shape
        return min(height, width) // 2, (width // 2, height // 2)
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit a circle to the contour
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    
    return int(radius), (int(x), int(y))

def calculate_heliographic_position(x, y, center_x, center_y, radius):
    """
    Calculate the heliographic position (latitude, longitude) of a point on the solar disk
    """
    # Normalize coordinates to [-1, 1] range relative to disk center
    x_norm = (x - center_x) / radius
    y_norm = (y - center_y) / radius
    
    # Distance from center in normalized coordinates
    rho = math.sqrt(x_norm**2 + y_norm**2)
    
    # If point is outside the disk, constrain to edge
    if rho > 1:
        rho = 1
    
    # Calculate heliographic latitude and longitude
    # Simplified calculation assuming perfect orientation
    lat = math.asin(y_norm) * 180 / math.pi  # latitude in degrees
    lon = math.asin(x_norm / math.cos(math.asin(y_norm))) * 180 / math.pi  # longitude in degrees
    
    return lat, lon

def adaptive_clustering(predictions, image_shape):
    """
    Perform adaptive clustering based on image scale and physics
    """
    # If no points to cluster, return empty labels
    if len(predictions) == 0:
        return np.array([])
    
    # Extract coordinates for clustering
    coordinates = np.array([[p['x'], p['y']] for p in predictions])
    
    # Estimate solar disk radius for scaling
    height, width = image_shape[:2]
    estimated_radius = min(height, width) / 2
    
    # Set eps parameter adaptively based on image size
    # For sunspot grouping, the typical separation is around 3-10 degrees heliographic
    # Here we use a simple heuristic of 5% of radius
    eps = max(estimated_radius * 0.05, 30)  # minimum 30 pixels for small images
    
    # Perform clustering
    clustering = DBSCAN(eps=eps, min_samples=1).fit(coordinates)
    
    return clustering.labels_

def refine_groups_by_physics(predictions, labels, disk_center, disk_radius):
    """
    Refine sunspot groups using solar physics principles
    """
    # If no spots, return empty labels
    if len(predictions) == 0:
        return np.array([])
    
    # Get unique group IDs
    unique_groups = set(labels)
    if -1 in unique_groups:
        unique_groups.remove(-1)
    
    # Compute heliographic positions for each spot
    helio_positions = []
    for prediction in predictions:
        x, y = prediction['x'], prediction['y']
        lat, lon = calculate_heliographic_position(x, y, disk_center[0], disk_center[1], disk_radius)
        helio_positions.append((lat, lon))
    
    # Create a mapping for new group assignments
    new_labels = labels.copy()
    
    # Apply physical rules to adjust groups
    # Rule: Spots within 10 degrees heliographic should be in same group
    for i in range(len(predictions)):
        lat1, lon1 = helio_positions[i]
        for j in range(i+1, len(predictions)):
            lat2, lon2 = helio_positions[j]
            
            # Calculate angular separation in degrees
            delta_lat = lat2 - lat1
            delta_lon = lon2 - lon1
            angular_dist = math.sqrt(delta_lat**2 + delta_lon**2)
            
            # If spots are close in heliographic coordinates
            # but in different groups, merge the groups
            if angular_dist < 10 and labels[i] != labels[j]:
                # Use the smaller group ID
                old_group = max(labels[i], labels[j])
                new_group = min(labels[i], labels[j])
                
                # Update all spots in the old group to the new group
                for k in range(len(new_labels)):
                    if new_labels[k] == old_group:
                        new_labels[k] = new_group
    
    return new_labels

def calculate_sunspot_number(predictions, image_shape):
    """
    Calculate the International Sunspot Number (ISN) based on SILSO method.
    Uses DBSCAN clustering to identify sunspot groups.
    R = k(10g + s)
    Where:
    - k is the observer factor (1.0 for this implementation)
    - g is the number of sunspot groups
    - s is the total number of individual sunspots
    """
    # If no sunspots, return 0
    if len(predictions) == 0:
        return 0, 0, 0, np.array([])
    
    # Get disk info for physical constraints
    estimated_radius, disk_center = estimate_solar_radius(np.zeros(image_shape, dtype=np.uint8))
    
    # Perform adaptive clustering
    labels = adaptive_clustering(predictions, image_shape)
    
    # Refine groups based on physics
    refined_labels = refine_groups_by_physics(predictions, labels, disk_center, estimated_radius)
    
    # Get the number of groups
    unique_labels = set(refined_labels)
    if -1 in unique_labels:  # Remove noise label if present
        unique_labels.remove(-1)
    num_groups = len(unique_labels)
    
    # Count total spots
    num_spots = len(predictions)
    
    # Calculate spots per group for analysis
    spots_per_group = {}
    for label in refined_labels:
        if label != -1:
            spots_per_group[label] = spots_per_group.get(label, 0) + 1
    
    # Wolf number calculation (k=1)
    k = 1.0
    wolf_number = k * (10 * num_groups + num_spots)
    
    return wolf_number, num_groups, num_spots, refined_labels

def visualize_predictions(image, predictions, labels=None):
    """
    Visualize the predictions on the image with rectangles around sunspots
    Color-coded by group
    """
    # Make a copy of the image to draw on
    img_with_labels = image.copy()
    
    # Sort predictions by position for consistent numbering
    # But keep original indices for label mapping
    sorted_with_indices = [(i, p) for i, p in enumerate(predictions)]
    sorted_with_indices.sort(key=lambda x: (x[1]["y"], x[1]["x"]))
    
    # If no labels provided, use default coloring
    if labels is None or len(labels) == 0:
        labels = np.zeros(len(predictions), dtype=int)
    
    # Assign colors to different groups
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(unique_labels), 1)))
    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    # Get solar disk info for drawing
    disk_radius, disk_center = estimate_solar_radius(image)
    
    # Draw solar disk outline
    cv2.circle(img_with_labels, disk_center, disk_radius, (200, 200, 200), 2)
    
    # Draw grid lines for reference
    for angle in range(0, 360, 30):  # Every 30 degrees longitude
        radian = math.radians(angle)
        end_x = int(disk_center[0] + disk_radius * math.cos(radian))
        end_y = int(disk_center[1] + disk_radius * math.sin(radian))
        cv2.line(img_with_labels, disk_center, (end_x, end_y), (150, 150, 150), 1)
    
    for radius_pct in [0.33, 0.66]:  # Latitude circles at approx 30 and 60 degrees
        cv2.circle(img_with_labels, disk_center, int(disk_radius * radius_pct), (150, 150, 150), 1)
    
    # Draw group outlines first
    for label in unique_labels:
        if label == -1:  # Skip noise
            continue
            
        # Get all points in this group
        group_points = []
        for i, pred in enumerate(predictions):
            if labels[i] == label:
                group_points.append((int(pred["x"]), int(pred["y"])))
        
        if len(group_points) > 2:
            # Compute convex hull around the group
            hull = cv2.convexHull(np.array(group_points))
            
            # Draw hull with group color
            color = label_to_color[label]
            # Convert from 0-1 RGB to 0-255 BGR for OpenCV
            hull_color = (int(color[2]*200), int(color[1]*200), int(color[0]*200))
            cv2.drawContours(img_with_labels, [hull], 0, hull_color, 2)
    
    # Draw rectangles and labels for each spot
    for idx, (orig_idx, prediction) in enumerate(sorted_with_indices, 1):
        x = prediction["x"]
        y = prediction["y"]
        
        # Get width and height if available, or use default size
        width = prediction.get("width", 20)
        height = prediction.get("height", 20)
        
        # Calculate rectangle coordinates
        x1 = int(x - width/2)
        y1 = int(y - height/2)
        x2 = int(x + width/2)
        y2 = int(y + height/2)
        
        # Get color for this group
        label = labels[orig_idx]
        color = label_to_color[label]
        # Convert from 0-1 RGB to 0-255 BGR for OpenCV
        box_color = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
        
        # Draw rectangle
        cv2.rectangle(img_with_labels, (x1, y1), (x2, y2), box_color, 2)
        
        # Add the number label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(idx)
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        
        # Position the text at the top-left of the rectangle
        text_x = x1
        text_y = y1 - 10 if y1 > 20 else y1 + 20
        
        # Draw white background for text
        cv2.rectangle(img_with_labels, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (255, 255, 255), 
                     -1)
        
        cv2.putText(img_with_labels, text, (text_x, text_y), 
                   font, 0.7, (0, 0, 0), 2)
        
        # Add heliographic coordinates
        lat, lon = calculate_heliographic_position(x, y, disk_center[0], disk_center[1], disk_radius)
        coord_text = f"({lat:.0f}°,{lon:.0f}°)"
        coord_size = cv2.getTextSize(coord_text, font, 0.5, 1)[0]
        
        coord_x = x1
        coord_y = y2 + 15
        
        cv2.rectangle(img_with_labels, 
                     (coord_x - 5, coord_y - coord_size[1] - 5),
                     (coord_x + coord_size[0] + 5, coord_y + 5),
                     (255, 255, 255), 
                     -1)
        
        cv2.putText(img_with_labels, coord_text, (coord_x, coord_y), 
                   font, 0.5, (0, 0, 0), 1)
    
    return img_with_labels

def get_first_image_from_sample_dir():
    """
    Get the first image from the sample_images directory
    """
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")
    if not os.path.exists(sample_dir):
        return None
    
    for file in os.listdir(sample_dir):
        file_path = os.path.join(sample_dir, file)
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            return file_path
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Run Sunspot detection model on an image")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap threshold (default: 0.5)")
    parser.add_argument("--preprocess", action="store_true", help="Apply preprocessing to enhance sunspot detection")
    args = parser.parse_args()
    
    if not args.image:
        args.image = get_first_image_from_sample_dir()
        if not args.image:
            print("Error: No image provided and no images found in sample_images directory")
            return
        print(f"Using image from sample directory: {args.image}")
    
    print("Loading Roboflow model...")
    api_key = "8ZF2l50a2MVTg0janXBE"
    if not api_key:
        print("Please set your ROBOFLOW_API_KEY environment variable")
        print("You can get your API key from your Roboflow account")
        return
    
    rf = Roboflow(api_key=api_key)
    
    workspace = rf.workspace()
    model = workspace.project("sunspot-nxzuy-gwbd7").version(1).model
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found")
        return
    
    # Load the image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    
    # Apply preprocessing if requested
    if args.preprocess:
        print("Applying image preprocessing...")
        preprocessed, equalized = preprocess_image(image)
        
        # Show preprocessing results
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(equalized, cmap='gray')
        plt.title("Contrast Enhanced")
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(preprocessed, cmap='gray')
        plt.title("Preprocessed Image")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    print(f"Running inference on {args.image}...")
    # Run inference
    predictions = model.predict(args.image, confidence=args.confidence, overlap=args.overlap).json()
    
    # Count spots and calculate sunspot number
    sunspot_num, num_groups, num_spots, labels = calculate_sunspot_number(predictions['predictions'], image.shape)
    
    # Print results
    print(f"Found {num_spots} sunspots in {num_groups} groups")
    print(f"Calculated sunspot number: {sunspot_num:.1f}")
    
    # Analyze group distribution
    if num_spots > 0:
        spots_per_group = {}
        for label in labels:
            if label != -1:
                spots_per_group[label] = spots_per_group.get(label, 0) + 1
        
        print("Group distribution:")
        for group_id, count in spots_per_group.items():
            print(f"  Group {group_id+1}: {count} spots")
    
    # Visualize the results
    img_with_labels = visualize_predictions(image, predictions['predictions'], labels)
    
    # Display the image with predictions
    plt.figure(figsize=(12, 12))
    plt.imshow(img_with_labels)
    plt.axis('off')
    plt.title(f"Sunspot Number: {sunspot_num:.1f} ({num_groups} groups, {num_spots} spots)", fontsize=14)
    plt.tight_layout()
    
    # Show the image
    plt.show()

if __name__ == "__main__":
    main()