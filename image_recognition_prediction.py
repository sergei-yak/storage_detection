import os
import glob
from ultralytics import YOLOWorld
from PIL import Image
import json

# Define the YOLO model path and directories
MODEL_PATH = "yolov8s-world.pt"
INPUT_DIR = 'auction_images'
OUTPUT_DIR = 'auction_images/predictions'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variable to store predictions
predicted_items = {}

def load_json(json_file):
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    return {}

auction_results = load_json('auction_results.json')

def load_model(model_path):
    """Load the YOLO model from the specified path."""
    return YOLOWorld(model_path)

def get_image_files(input_dir, extensions=["*.jpg", "*.jpeg", "*.png"]):
    """Get a list of image files from the specified directory."""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    return files

def is_valid_image(image_path):
    """Check if the given image path is a valid image."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def predict_and_save(model, image_path, output_dir):
    """Run prediction on an image and save the output to the specified directory."""
    if not is_valid_image(image_path):
        print(f"WARNING! Skipping invalid image: {image_path}")
        return None, None, None

    prediction_results = model.predict(image_path)
    base_filename = os.path.basename(image_path)
    prediction_filename = f"pred_{base_filename}"
    output_path = os.path.join(output_dir, prediction_filename)

    for result in prediction_results:
        result.save(output_path)

    return base_filename, prediction_filename, prediction_results

def parse_predictions(prediction_results, model):
    """Extract class names and confidence scores from prediction results."""
    prediction_data = {}
    for result in prediction_results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            class_name = model.names[class_id]
            if class_name not in prediction_data:
                prediction_data[class_name] = []
            prediction_data[class_name].append(confidence)
    return prediction_data

def process_images(model, input_dir, output_dir):
    """Iterate through all images in the input directory and store results."""
    image_files = get_image_files(input_dir)
    global predicted_items

    # we need this to add the auction_id to predicted_items
    auction_lookup = {
        item['image_name']: item['auction_id']
        for images in auction_results.values()
        for item in images
    }

    for image_file in image_files:
        original_filename, prediction_filename, results = predict_and_save(model, image_file, output_dir)
        if original_filename is None:
            continue
        prediction_data = parse_predictions(results, model)

        # matches auction_id to original_filename
        auction_id = auction_lookup.get(original_filename)

        predicted_items[original_filename] = {
            #"original_filename": original_filename, # deleted, because it is a duplicate and OpenAI limit
            "prediction_filename": prediction_filename,
            "predicted_data": prediction_data,
            "auction_id": auction_id
        }

def main():
    """Main function to load model, process images, and store predictions."""
    model = load_model(MODEL_PATH)
    process_images(model, INPUT_DIR, OUTPUT_DIR)

    # Output the final dictionary with all detected objects
    for item, data in predicted_items.items():
        print(f"Original File: {item}, Predicted File: {data['prediction_filename']}") # replaced {data['original_filename']} with item
        print("Detected Objects:")
        for class_name, confidences in data['predicted_data'].items():
            avg_confidence = sum(confidences) / len(confidences)
            print(f"  {class_name}: {len(confidences)} instances with avg confidence {avg_confidence:.2f}")

    return predicted_items

# Run the main function only if executed directly
if __name__ == "__main__":
    predicted_items = main()
    #save the output to a json file
    with open('predicted_items.json', 'w') as f:
        json.dump(predicted_items, f, indent=4)
