import os
from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np

def visualize_predictions(image, predictions):
    """
    Visualize the predictions on the image with NASA-style numbered labels
    """
    # Make a copy of the image to draw on
    img_with_labels = image.copy()
    
    # Sort predictions by position for consistent numbering
    predictions = sorted(predictions, key=lambda p: (p["y"], p["x"]))
    
    # Draw numbered labels
    for i, prediction in enumerate(predictions, 1):
        x = prediction["x"]
        y = prediction["y"]
        
        # Draw a small circle at the center of the spot
        cv2.circle(img_with_labels, (int(x), int(y)), 5, (255, 0, 0), -1)
        
        # Add the number label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(i)
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        
        # Position the text slightly offset from the center
        text_x = int(x) + 10
        text_y = int(y) - 10
        
        # Draw white background for text
        cv2.rectangle(img_with_labels, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (255, 255, 255), 
                     -1)
        
        # Draw the number
        cv2.putText(img_with_labels, text, (text_x, text_y), 
                   font, 0.7, (255, 0, 0), 2)
    
    return img_with_labels

def calculate_sunspot_number(num_spots):
    """
    Calculate the International Sunspot Number (ISN) based on SILSO method.
    Simple version - assumes each spot is its own group.
    R = k(10g + s)
    """
    # For simplicity, consider each spot as its own group
    groups = num_spots
    
    # Wolf number calculation (k=1)
    k = 1
    wolf_number = k * (10 * groups + num_spots)
    
    return wolf_number

def main():
    parser = argparse.ArgumentParser(description="Run Sunspot detection model on an image")
    parser.add_argument("--image", type=str, help="Path to the image file", required=True)
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap threshold (default: 0.5)")
    parser.add_argument("--save", action="store_true", help="Save the output image")
    args = parser.parse_args()
    
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
    
    # Count spots
    num_spots = len(predictions['predictions'])
    sunspot_number = calculate_sunspot_number(num_spots)
    
    # Print results
    print(f"Found {num_spots} sunspots")
    print(f"Estimated International Sunspot Number: {sunspot_number}")
    
    # Visualize the results
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    
    img_with_labels = visualize_predictions(image, predictions['predictions'])
    
    # Display the image with predictions - show entire solar disk
    plt.figure(figsize=(12, 12))
    plt.imshow(img_with_labels)
    plt.axis('off')
    plt.title(f"Sunspot Detection: {num_spots} spots\nEstimated International Sunspot Number: {sunspot_number}", fontsize=14)
    plt.tight_layout()
    
    # Save the output image if requested
    if args.save:
        output_path = os.path.splitext(args.image)[0] + "_analyzed.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved analysis to {output_path}")
    
    # Show the image
    plt.show()

if __name__ == "__main__":
    main()