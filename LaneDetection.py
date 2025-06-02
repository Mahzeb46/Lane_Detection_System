# Import necessary libraries for image processing, GUI creation, and file handling
import cv2                      # OpenCV for computer vision tasks like image processing
import numpy as np              # NumPy for numerical computations and array handling
import tkinter as tk            # Tkinter for creating the GUI window and widgets
from PIL import Image, ImageTk  # PIL for handling images within Tkinter
import os                       # OS module for directory operations

# Specify the directory containing lane detection images
DATASET_FOLDER = r"C:\Users\PMLS\Desktop\DIP_Project\Dataset_Lane"

# Define the main application class for the lane detection GUI
class LaneDetectionApp:
    def __init__(self, root):
        """
        Initialize the application window, parameters, and GUI elements.
        """
        # Set the main window reference
        self.root = root
        self.root.title("🚗 Lane Detection - DIP Project")  # Set window title
        self.root.geometry("1280x720")                     # Set window size

        # Load all image file paths from the dataset folder with image file extensions
        self.image_files = sorted([
            os.path.join(DATASET_FOLDER, f)
            for f in os.listdir(DATASET_FOLDER)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))  # Supported formats
        ])

        # Raise an error if no images found in the folder
        if not self.image_files:
            raise FileNotFoundError("No images found in Dataset_Lane folder.")

        # Initialize index to track the current image being processed
        self.image_index = 0

        # Initialize adjustable parameters using Tkinter IntVar to link with sliders
        self.params = {
            "Canny Low": tk.IntVar(value=100),       # Lower threshold for Canny Edge Detection
            "Canny High": tk.IntVar(value=200),      # Upper threshold for Canny Edge Detection
            "Gaussian Kernel": tk.IntVar(value=5),   # Kernel size for Gaussian Blur (must be odd)
            "Hough Threshold": tk.IntVar(value=50),  # Minimum number of intersections for Hough Line
            "Min Line Length": tk.IntVar(value=40),  # Minimum line length for Hough transform
            "Max Line Gap": tk.IntVar(value=50),     # Maximum allowed gap between points on a line
            "ROI Height": tk.IntVar(value=100)       # Height of the region of interest mask from bottom
        }

        # Create all GUI components
        self.create_widgets()

        # Process the first image to display initial output
        self.process_image()

    def create_widgets(self):
        """
        Set up GUI layout: image preview canvases, sliders, and navigation buttons.
        """
        # Create a frame to display original, overlay, and binary images
        self.image_frame = tk.LabelFrame(self.root, text="Image Preview", padx=10, pady=10)
        self.image_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Create Tkinter Labels (as image canvases)
        self.canvas_original = tk.Label(self.image_frame)  # Original image display
        self.canvas_overlay = tk.Label(self.image_frame)   # Overlay image display
        self.canvas_binary = tk.Label(self.image_frame)    # Binary output display

        # Position image canvases in a grid layout
        self.canvas_original.grid(row=0, column=0, padx=10)
        self.canvas_overlay.grid(row=0, column=1, padx=10)
        self.canvas_binary.grid(row=0, column=2, padx=10)

        # Add text labels beneath each image display
        tk.Label(self.image_frame, text="Original").grid(row=1, column=0)
        tk.Label(self.image_frame, text="Lane Overlay").grid(row=1, column=1)
        tk.Label(self.image_frame, text="Binary Output").grid(row=1, column=2)

        # Create a frame for slider controls
        controls = tk.LabelFrame(self.root, text="Controls", padx=10, pady=10)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Group sliders into rows and columns
        for i, row_keys in enumerate([("Canny Low", "Canny High", "Gaussian Kernel"),
                                      ("Hough Threshold", "Min Line Length", "Max Line Gap"),
                                      ("ROI Height",)]):
            row = tk.Frame(controls)
            row.pack(fill=tk.X)
            for j, key in enumerate(row_keys):
                self.create_slider(row, key, j, max_val=15 if key == "Gaussian Kernel" else 255)

        # Create navigation and process buttons
        nav = tk.Frame(self.root)
        nav.pack(pady=10)
        tk.Button(nav, text="<< Prev", command=self.prev_image, width=10).pack(side=tk.LEFT, padx=10)
        tk.Button(nav, text="Process", command=self.process_image, width=10).pack(side=tk.LEFT, padx=10)
        tk.Button(nav, text="Next >>", command=self.next_image, width=10).pack(side=tk.LEFT, padx=10)

    def create_slider(self, parent, name, column, max_val=255):
        """
        Create a slider and label for an adjustable parameter.
        """
        tk.Label(parent, text=name).grid(row=0, column=column, padx=5)
        tk.Scale(parent, from_=0, to=max_val, orient=tk.HORIZONTAL,
                 variable=self.params[name], length=200).grid(row=1, column=column, padx=5)

    def get_param(self, key):
        """
        Get parameter value from slider. Ensure Gaussian Kernel is odd.
        """
        val = self.params[key].get()
        # Force Gaussian kernel to be odd
        return val + 1 if key == "Gaussian Kernel" and val % 2 == 0 else val

    def prev_image(self):
        """Switch to the previous image in the list."""
        self.image_index = (self.image_index - 1) % len(self.image_files)
        self.process_image()

    def next_image(self):
        """Switch to the next image in the list."""
        self.image_index = (self.image_index + 1) % len(self.image_files)
        self.process_image()

    def process_image(self):
        """Main image processing pipeline: color filtering, edge detection, ROI, and line detection."""
        image_path = self.image_files[self.image_index]
        original = cv2.imread(image_path)
        if original is None:
            return
        original = cv2.resize(original, (640, 360))  # Resize to fixed size

        # Convert to HLS color space for better color filtering
        #HLS (Hue Lightness Saturation) is better for color-based lane detection.


        hls = cv2.cvtColor(original, cv2.COLOR_BGR2HLS)

        # Create masks for detecting white and yellow lane markings
        white = cv2.inRange(hls, np.array([0, 200, 0]), np.array([255, 255, 255]))
        yellow = cv2.inRange(hls, np.array([15, 30, 115]), np.array([35, 204, 255]))

        # Combine white and yellow masks
        combined_mask = cv2.bitwise_or(white, yellow)

        # Apply mask to original image
        masked = cv2.bitwise_and(original, original, mask=combined_mask)

        # Convert masked image to grayscale
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (self.get_param("Gaussian Kernel"), self.get_param("Gaussian Kernel")), 0)

        # Apply Canny edge detection on the blurred grayscale image
        edges = cv2.Canny(blur, self.get_param("Canny Low"), self.get_param("Canny High"))

        # Define region of interest (ROI) mask
        height, width = edges.shape
        roi_height = self.get_param("ROI Height")
        roi_mask = np.zeros_like(edges)

        # Define ROI polygon points
        polygon = np.array([[
            (int(0.05 * width), height), (int(0.4 * width), height - roi_height),
            (int(0.6 * width), height - roi_height), (int(0.95 * width), height)
        ]], dtype=np.int32)

        # Fill ROI polygon in mask with white
        cv2.fillPoly(roi_mask, polygon, 255)

        # Apply ROI mask to the edge image
        roi_edges = cv2.bitwise_and(edges, roi_mask)

        # Detect line segments using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            roi_edges, 1, np.pi / 180,
            self.get_param("Hough Threshold"),
            minLineLength=self.get_param("Min Line Length"),
            maxLineGap=self.get_param("Max Line Gap")
        )

        # Create empty images to draw detected lines
        line_img = np.zeros_like(original)
        binary_mask = np.zeros_like(original)

        # Process each detected line segment
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                slope = dy / (dx + 1e-5)  # Avoid division by zero

                # Color lines based on their slope
                if abs(slope) > 0.4:
                    color = (0, 255, 255)  # Yellow for steep slopes
                    width_px = 3
                elif abs(slope) > 0.1:
                    color = (0, 0, 255)    # Red for moderate slopes
                    width_px = 2
                else:
                    continue  # Skip nearly horizontal lines

                # Draw line on overlay and binary mask
                cv2.line(line_img, (x1, y1), (x2, y2), color, width_px)
                cv2.line(binary_mask, (x1, y1), (x2, y2), (255, 255, 255), width_px)

        # Combine original image with detected lines
        overlay = cv2.addWeighted(original, 0.8, line_img, 1, 0)

        # Update canvases with new images
        self.update_canvas(original, self.canvas_original)
        self.update_canvas(overlay, self.canvas_overlay)
        self.update_canvas(binary_mask, self.canvas_binary)

    def update_canvas(self, img, canvas):
        """Display an image in the given Tkinter canvas."""
        img_rgb = cv2.cvtColor(cv2.resize(img, (400, 240)), cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        canvas.configure(image=imgtk)
        canvas.image = imgtk  # Prevent image garbage collection

# Start the Tkinter application
if __name__ == "__main__":
    root = tk.Tk()
    app = LaneDetectionApp(root)
    root.mainloop()
