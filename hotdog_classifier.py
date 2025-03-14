from neural_network import NeuralNetwork
from image_processor import ImageProcessor
import os
import shutil

MODEL_FILE = 'hotdog_model.json'

def create_folders():
    """Create all necessary folders if they don't exist"""
    folders = [
        'training_data/hotdog',
        'training_data/not_hotdog',
        'test_images'
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print("\nFolder structure created:")
    print("- training_data/hotdog/    (for confirmed hotdog images)")
    print("- training_data/not_hotdog/ (for confirmed non-hotdog images)")
    print("- test_images/             (place new images here for testing)")
    print("\nImage requirements:")
    print("- Format: .jpg, .jpeg, or .png")
    print("- Will be converted to 24x24 RGB")

def move_to_training(image_name, is_hotdog):
    """Move an image from test_images to the appropriate training folder"""
    source = os.path.join('test_images', image_name)
    target_dir = 'training_data/hotdog' if is_hotdog else 'training_data/not_hotdog'
    target = os.path.join(target_dir, image_name)
    
    try:
        shutil.move(source, target)
        print(f"\nMoved {image_name} to {target_dir}/")
        return True
    except Exception as e:
        print(f"\nError moving file: {str(e)}")
        return False

def list_test_images():
    """List all images in the test_images folder"""
    if not os.path.exists('test_images'):
        return []
    
    return [f for f in os.listdir('test_images') 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def train_hotdog_classifier(force_retrain=False):
    """Train the neural network on hotdog/not-hotdog images"""
    # Check if we have a saved model
    if not force_retrain and os.path.exists(MODEL_FILE):
        print("\nLoading existing model...")
        return NeuralNetwork.load_model(MODEL_FILE)
    
    # Initialize image processor (24x24 pixels * 3 channels = 1728 inputs)
    processor = ImageProcessor(target_size=(24, 24))
    
    # Process hotdog images
    print("\nProcessing training images...")
    hotdog_images, hotdog_labels = processor.process_directory('training_data/hotdog')
    not_hotdog_images, not_hotdog_labels = processor.process_directory('training_data/not_hotdog')
    
    # Combine datasets
    training_data = hotdog_images + not_hotdog_images
    targets = hotdog_labels + not_hotdog_labels
    
    if not training_data:
        print("No training images found! Please add images to the training_data folders.")
        return None
    
    print(f"\nTraining on {len(training_data)} images...")
    print(f"- Hotdogs: {len(hotdog_images)}")
    print(f"- Not hotdogs: {len(not_hotdog_images)}")
    
    # Simple but effective network
    nn = NeuralNetwork([1728, 32, 1])
    nn.train(training_data, targets, learning_rate=0.001, epochs=300)
    
    # Save the trained model
    nn.save_model(MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    
    return nn

def test_classifier(nn, image_name):
    """Test the classifier on a single image from test_images folder"""
    image_path = os.path.join('test_images', image_name)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_name}")
        return None
    
    processor = ImageProcessor(target_size=(24, 24))
    processed_image = processor.process_image(image_path)
    
    if processed_image is not None:
        result = nn.predict(processed_image)[0]
        print(f"\nResults for: {image_name}")
        print(f"Hotdog probability: {result:.2%}")
        print(f"Classification: {'Hotdog' if result > 0.5 else 'Not Hotdog'}")
        return result
    return None

if __name__ == "__main__":
    # Create folders
    create_folders()
    
    # Train or load the network
    nn = train_hotdog_classifier()
    
    if nn is not None:
        print("\nModel ready! You can now test images.")
        
        while True:
            print("\nOptions:")
            print("1. List test images")
            print("2. Test an image")
            print("3. Retrain network")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == "4":
                break
            elif choice == "1":
                test_images = list_test_images()
                if test_images:
                    print("\nAvailable test images:")
                    for i, img in enumerate(test_images, 1):
                        print(f"{i}. {img}")
                else:
                    print("\nNo images found in test_images folder!")
            elif choice == "2":
                test_images = list_test_images()
                if not test_images:
                    print("\nNo images found in test_images folder!")
                    continue
                
                print("\nAvailable test images:")
                for i, img in enumerate(test_images, 1):
                    print(f"{i}. {img}")
                
                try:
                    img_num = int(input("\nEnter image number to test: ")) - 1
                    if 0 <= img_num < len(test_images):
                        image_name = test_images[img_num]
                        result = test_classifier(nn, image_name)
                        
                        if result is not None:
                            while True:
                                feedback = input("\nWas this classification correct? (y/n): ").lower()
                                if feedback in ['y', 'n']:
                                    break
                                print("Please enter 'y' or 'n'")
                            
                            if feedback == 'y':
                                # Move to appropriate training folder
                                is_hotdog = result > 0.5
                                if move_to_training(image_name, is_hotdog):
                                    print("Image added to training data!")
                                    print("You can use option 3 to retrain the network with this new data.")
                            else:
                                # Move to opposite folder
                                is_hotdog = result <= 0.5
                                if move_to_training(image_name, is_hotdog):
                                    print("Image added to training data with corrected label!")
                                    print("You can use option 3 to retrain the network with this new data.")
                    else:
                        print("Invalid image number!")
                except ValueError:
                    print("Please enter a valid number!")
            elif choice == "3":
                print("\nRetraining network with all available data...")
                nn = train_hotdog_classifier(force_retrain=True)
            else:
                print("Invalid choice. Please enter 1-4.")
