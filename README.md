# Sunspot Detection Model - Local Inference

This project allows you to run the Roboflow sunspot detection model (`sunspot-nxzuy-gwbd7`) locally.

## Requirements

- Python 3.7+
- Virtual environment (recommended)
- Roboflow API key

## Setup

1. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install roboflow
   ```

3. Set your Roboflow API key as an environment variable:
   ```
   export ROBOFLOW_API_KEY="your_api_key_here"  # On Windows: set ROBOFLOW_API_KEY=your_api_key_here
   ```

   You can get your API key from your Roboflow account dashboard.

## Running the Model

Use the `run_sunspot_model.py` script to perform inference on local images:

```
python run_sunspot_model.py --image path/to/your/image.jpg
```

### Optional Arguments

- `--confidence`: Set the confidence threshold (default: 0.5)
- `--overlap`: Set the overlap threshold for non-maximum suppression (default: 0.5)

Example:
```
python run_sunspot_model.py --image sun_image.jpg --confidence 0.7 --overlap 0.4
```

## Output

The script will:
1. Run inference on the provided image
2. Display the image with bounding boxes around detected sunspots
3. Save the annotated image with the suffix "_predictions.jpg"
4. Print detection results to the console

## Troubleshooting

If you encounter any issues:

1. Verify that your API key is correct and properly set
2. Ensure your image file exists and is in a readable format
3. Try adjusting the confidence threshold if you're getting too few or too many detections 