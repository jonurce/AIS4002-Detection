from PIL import Image
import os


def resize_image(input_path, output_path, new_width, new_height):

    try:
        # Open the image
        with Image.open(input_path) as img:
            # Resize image while maintaining aspect ratio
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save the resized image
            img.save(output_path, quality=95)
            print(f"Image successfully resized and saved to {output_path}")

    except Exception as e:
        print(f"Error resizing image: {str(e)}")


def main():
    # Example usage
    name = 'test_4_crowded'
    input_image = f"Original/{name}.jpg"  # Replace with your image path
    output_image = f"{name}.jpg"
    width = 1280
    height = 720

    # Check if input file exists
    if not os.path.exists(input_image):
        print(f"Input image {input_image} not found")
        return

    resize_image(input_image, output_image, width, height)


if __name__ == "__main__":
    main()