import argparse
import cv2
import os
from image_processing import process_image

def save_output_image(warped, output_path):
    """Saves the scanned document image to the specified path."""
    cv2.imwrite(output_path, warped)
    print(f"Scanned document saved to {output_path}")

if __name__ == "__main__":
    # Argument parser
    ap = argparse.ArgumentParser(description="Process and scan a document from an image.")
    ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    ap.add_argument("-o", "--output", help="Path to save the scanned image (optional, defaults to 'scanned/')")
    args = vars(ap.parse_args())

    try:
        # Process the image
        warped = process_image(args["image"])

        # Determine the output folder (default to "scanned/")
        output_folder = args["output"] if args["output"] else "scanned/"
        
        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(args["image"]))[0]

        # Create the output file path
        output_path = os.path.join(output_folder, f"{filename}_scanned.jpg")

        # Save the output image
        save_output_image(warped, output_path)

    except Exception as e:
        print(f"Error processing image: {str(e)}")
