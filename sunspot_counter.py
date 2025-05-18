import os
from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.cluster import DBSCAN

def visualize_predictions(image, predictions, sunspot_info=None):
    """
    Visualize the predictions on the image with rectangles around sunspots
    """
    # Make a copy of the image to draw on
    img_with_labels = image.copy()
    
    # Sort predictions by position for consistent numbering
    predictions = sorted(predictions, key=lambda p: (p["y"], p["x"]))
    
    # Extract coordinates for clustering if sunspot_info is not provided
    if sunspot_info is None:
        coordinates = np.array([[p['x'], p['y']] for p in predictions])
        clustering = DBSCAN(eps=100, min_samples=1).fit(coordinates)
        labels = clustering.labels_
    else:
        # Use provided coordinates and labels (if any)
        coordinates = np.array([[p['x'], p['y']] for p in predictions])
        clustering = DBSCAN(eps=100, min_samples=1).fit(coordinates)
        labels = clustering.labels_
    
    # Assign colors to different groups
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Draw rectangles and labels
    for i, (prediction, label) in enumerate(zip(predictions, labels), 1):
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
        color = label_to_color[label]
        # Convert from 0-1 RGB to 0-255 BGR for OpenCV
        box_color = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
        
        # Draw rectangle
        cv2.rectangle(img_with_labels, (x1, y1), (x2, y2), box_color, 2)
        
        # Add the number label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(i)
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
        
        # Draw the number
        cv2.putText(img_with_labels, text, (text_x, text_y), 
                   font, 0.7, (255, 0, 0), 2)
    
    return img_with_labels, labels

def calculate_sunspot_number(predictions):
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
        return 0, 0, 0
    
    # Extract coordinates for clustering
    coordinates = np.array([[p['x'], p['y']] for p in predictions])
    
    # Use DBSCAN to cluster sunspots into groups
    # The eps parameter determines the maximum distance between two points to be considered in the same group
    # This may need tuning based on typical image dimensions and sunspot distributions
    clustering = DBSCAN(eps=100, min_samples=1).fit(coordinates)
    
    # Get the number of groups (number of unique cluster labels, excluding noise which is -1)
    unique_labels = set(clustering.labels_)
    if -1 in unique_labels:  # Remove noise label if present
        unique_labels.remove(-1)
    num_groups = len(unique_labels)
    
    # Count total spots
    num_spots = len(predictions)
    
    # Wolf number calculation (k=1)
    k = 1.0
    wolf_number = k * (10 * num_groups + num_spots)
    
    return wolf_number, num_groups, num_spots

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
    args = parser.parse_args()
    
    # If no image is provided, use the first image from sample_images directory
    if not args.image:
        args.image = get_first_image_from_sample_dir()
        if not args.image:
            print("Error: No image provided and no images found in sample_images directory")
            return
        print(f"Using image from sample directory: {args.image}")
    
    print("Loading Roboflow model...")
    # Initialize Roboflow API
    api_key = "8ZF2l50a2MVTg0janXBE"
    if not api_key:
        print("Please set your ROBOFLOW_API_KEY environment variable")
        print("You can get your API key from your Roboflow account")
        return
    
    rf = Roboflow(api_key=api_key)
    
    # Load the project with the specific model ID
    workspace = rf.workspace()
    model = workspace.project("sunspot-nxzuy-gwbd7").version(1).model
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found")
        return
    
    print(f"Running inference on {args.image}...")
    # Run inference
    predictions = model.predict(args.image, confidence=args.confidence, overlap=args.overlap).json()
    
    # Count spots and calculate sunspot number
    sunspot_num, num_groups, num_spots = calculate_sunspot_number(predictions['predictions'])
    
    # Print results
    print(f"Found {num_spots} sunspots in {num_groups} groups")
    print(f"Calculated sunspot number: {sunspot_num:.1f}")
    
    # Visualize the results
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    
    img_with_labels, labels = visualize_predictions(image, predictions['predictions'])
    
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