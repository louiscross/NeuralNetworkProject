from PIL import Image
import os

class ImageProcessor:
    def __init__(self, target_size=(64, 64)):
        """Initialize with target size for all images"""
        self.target_size = target_size

    def process_image(self, image_path):
        """
        Process a single image:
        1. Load the image
        2. Resize to target size
        3. Extract RGB channels
        4. Normalize pixel values to [0,1]
        5. Apply basic image enhancement
        """
        try:
            # Load and resize image
            img = Image.open(image_path)
            
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Enhance image
            img = self._enhance_image(img)
            
            # Resize image
            img = img.resize(self.target_size)
            
            # Split into RGB channels and normalize
            r, g, b = img.split()
            r_values = [p / 255.0 for p in r.getdata()]
            g_values = [p / 255.0 for p in g.getdata()]
            b_values = [p / 255.0 for p in b.getdata()]
            
            # Combine all channels
            normalized_pixels = r_values + g_values + b_values
            
            return normalized_pixels
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def _enhance_image(self, img):
        """Apply basic image enhancement"""
        from PIL import ImageEnhance
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        # Enhance color
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.3)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        
        return img

    def process_directory(self, directory_path):
        """Process all images in a directory"""
        processed_images = []
        labels = []
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(directory_path, filename)
                processed_image = self.process_image(image_path)
                
                if processed_image is not None:
                    processed_images.append(processed_image)
                    # If filename contains 'hotdog', label as 1, else 0
                    labels.append([1.0] if 'hotdog' in filename.lower() else [0.0])
        
        return processed_images, labels
