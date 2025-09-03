"""
FIXED Iris Segmentation using Geodesic Active Contours (GAC)
Improved version with better boundary constraints and shape regularization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_erosion, binary_dilation
from skimage import measure, morphology, filters
from skimage.feature import canny
import warnings
warnings.filterwarnings('ignore')

class FixedIrisGACSegmentation:
    """
    Fixed Iris Segmentation with better boundary constraints
    """
    
    def __init__(self, alpha=0.4, beta=0.6, gamma=0.02, iterations=250, 
                 dt=0.05, sigma=1.2, reinit_freq=20):
        """
        Initialize FIXED GAC parameters - more conservative approach
        """
        self.alpha = alpha      # Higher curvature weight for smoothness
        self.beta = beta        # Higher edge attraction
        self.gamma = gamma      # LOWER balloon force to prevent over-expansion
        self.iterations = iterations
        self.dt = dt           # Smaller time step for stability
        self.sigma = sigma
        self.reinit_freq = reinit_freq
    
    def enhanced_preprocessing(self, image):
        """
        IMPROVED preprocessing focused on iris boundaries
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to [0, 1]
        gray = gray.astype(np.float64) / 255.0
        
        # Apply moderate histogram equalization (less aggressive)
        gray_uint8 = (gray * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Reduced clip limit
        enhanced = clahe.apply(gray_uint8)
        gray = enhanced.astype(np.float64) / 255.0
        
        # Light bilateral filtering
        gray_uint8 = (gray * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(gray_uint8, 5, 50, 50)  # Reduced parameters
        gray = filtered.astype(np.float64) / 255.0
        
        # Minimal Gaussian smoothing to preserve iris boundaries
        gray = gaussian_filter(gray, sigma=0.8)  # Reduced sigma
        
        return gray
    
    def improved_edge_stopping_function(self, image):
        """
        ENHANCED edge stopping function with better iris boundary detection
        """
        # Apply edge-preserving smoothing
        smoothed = gaussian_filter(image, sigma=0.8)
        
        # Compute gradients with different kernel sizes for multi-scale detection
        grad_x_3 = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_3 = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
        grad_x_5 = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=5)
        grad_y_5 = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=5)
        
        # Combine multi-scale gradients
        grad_x = 0.7 * grad_x_3 + 0.3 * grad_x_5
        grad_y = 0.7 * grad_y_3 + 0.3 * grad_y_5
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Enhanced Canny edge detection for iris boundaries
        edges = canny(smoothed, sigma=1.0, low_threshold=0.05, high_threshold=0.15)
        
        # STRONGER edge stopping function to better detect iris boundaries
        g = 1.0 / (1.0 + 5.0 * gradient_magnitude**2 + 2.0 * edges.astype(float))
        
        # Apply smoothing to edge stopping function
        g = gaussian_filter(g, sigma=0.5)
        
        return g, grad_x, grad_y
    
    def smart_pupil_iris_detection(self, image):
        """
        IMPROVED detection with better parameter tuning for iris boundaries
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Enhanced preprocessing for circle detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        blurred = cv2.GaussianBlur(enhanced, (11, 11), 2)
        
        # CONSERVATIVE pupil detection
        pupil_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=h//3,
            param1=60, param2=35, 
            minRadius=max(5, h//25), maxRadius=min(h//5, w//5)
        )
        
        # MORE CONSERVATIVE iris detection with stricter constraints
        iris_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=h//2,
            param1=40, param2=60,
            minRadius=max(h//6, w//6), maxRadius=min(h//3, w//3)  # Smaller max radius
        )
        
        # Default values (more conservative)
        pupil_center = (w//2, h//2)
        pupil_radius = min(h//15, w//15)  # Smaller default pupil
        iris_radius = min(h//5, w//5)    # Smaller default iris
        
        if pupil_circles is not None and len(pupil_circles[0]) > 0:
            pupil = np.round(pupil_circles[0, 0]).astype("int")
            pupil_center = (int(pupil[0]), int(pupil[1]))
            pupil_radius = int(pupil[2])
        
        if iris_circles is not None and len(iris_circles[0]) > 0:
            iris = np.round(iris_circles[0, 0]).astype("int")
            # Use pupil center but iris radius
            iris_radius = min(int(iris[2]), min(h//3, w//3))  # Cap the iris radius
        
        # Ensure iris radius is reasonable compared to pupil
        iris_radius = max(iris_radius, pupil_radius * 2.5)  # At least 2.5x pupil radius
        iris_radius = min(iris_radius, pupil_radius * 6)    # At most 6x pupil radius
        
        return pupil_center, pupil_radius, iris_radius
    
    def initialize_constrained_level_sets(self, image_shape, center, pupil_radius, iris_radius):
        """
        Initialize level sets with SHAPE CONSTRAINTS
        """
        h, w = image_shape
        y, x = np.ogrid[0:h, 0:w]
        center_x, center_y = center
        
        # Pupil level set with safety margins
        phi_pupil = np.sqrt((x - center_x)**2 + (y - center_y)**2) - (pupil_radius * 0.8)
        
        # Iris level set with conservative initialization
        phi_iris = np.sqrt((x - center_x)**2 + (y - center_y)**2) - (iris_radius * 0.9)
        
        return phi_pupil.astype(np.float64), phi_iris.astype(np.float64)
    
    def evolve_constrained_contours(self, image, phi_pupil, phi_iris, center, pupil_radius, iris_radius):
        """
        Evolve contours with SHAPE CONSTRAINTS to prevent over-segmentation
        """
        g, g_x, g_y = self.improved_edge_stopping_function(image)
        
        print("Starting constrained GAC evolution...")
        
        for i in range(self.iterations):
            # Evolve pupil (contracting)
            phi_pupil = self.evolve_constrained_contour(
                phi_pupil, g, g_x, g_y, center, pupil_radius, balloon_sign=-1, is_pupil=True
            )
            
            # Evolve iris (expanding with constraints)
            phi_iris = self.evolve_constrained_contour(
                phi_iris, g, g_x, g_y, center, iris_radius, balloon_sign=1, is_pupil=False
            )
            
            # More frequent reinitialization for stability
            if (i + 1) % self.reinit_freq == 0:
                phi_pupil = self.reinitialize_sdf(phi_pupil)
                phi_iris = self.reinitialize_sdf(phi_iris)
            
            if i % 50 == 0:
                print(f"Iteration {i}/{self.iterations}")
        
        print("Constrained GAC evolution completed!")
        return phi_pupil, phi_iris
    
    def evolve_constrained_contour(self, phi, g, g_x, g_y, center, max_radius, balloon_sign=1, is_pupil=True):
        """
        Evolve single contour with DISTANCE CONSTRAINTS
        """
        # Compute spatial derivatives
        phi_y, phi_x = np.gradient(phi)
        phi_norm = np.sqrt(phi_x**2 + phi_y**2 + 1e-10)
        
        # Compute curvature
        curvature = self.compute_curvature(phi)
        
        # Create distance constraint mask
        h, w = phi.shape
        y, x = np.ogrid[0:h, 0:w]
        center_x, center_y = center
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Constraint factors
        if is_pupil:
            # For pupil: prevent excessive shrinking
            constraint_factor = np.ones_like(phi)
            constraint_factor[distance_from_center < max_radius * 0.3] = 0.1
        else:
            # For iris: prevent excessive expansion beyond reasonable iris size
            max_allowed_radius = max_radius * 1.2  # Allow 20% expansion beyond detected radius
            constraint_factor = np.ones_like(phi)
            constraint_factor[distance_from_center > max_allowed_radius] = 0.1
        
        # GAC evolution components with constraints
        curvature_term = g * curvature * phi_norm
        advection_term = g_x * phi_x + g_y * phi_y
        balloon_term = balloon_sign * g * phi_norm * constraint_factor
        
        # Combined evolution with REDUCED balloon force
        dphi_dt = (self.alpha * curvature_term + 
                  self.beta * advection_term + 
                  self.gamma * balloon_term)
        
        # Update with smaller time step
        phi = phi + self.dt * dphi_dt
        
        return phi
    
    def compute_curvature(self, phi):
        """
        Robust curvature computation
        """
        # First derivatives
        phi_y, phi_x = np.gradient(phi)
        
        # Second derivatives
        phi_xx, phi_xy = np.gradient(phi_x)
        phi_yy, phi_yx = np.gradient(phi_y)
        
        # Compute curvature with numerical stability
        phi_norm_sq = phi_x**2 + phi_y**2
        phi_norm = np.sqrt(phi_norm_sq + 1e-12)
        
        curvature = ((phi_xx * phi_y**2 - 2 * phi_xy * phi_x * phi_y + phi_yy * phi_x**2) / 
                    (phi_norm_sq * phi_norm + 1e-12))
        
        # Stronger clamping for stability
        curvature = np.clip(curvature, -5, 5)
        
        return curvature
    
    def reinitialize_sdf(self, phi):
        """
        Stable reinitialization
        """
        zero_level = phi > 0
        pos_dist = distance_transform_edt(zero_level)
        neg_dist = distance_transform_edt(~zero_level)
        sdf = pos_dist - neg_dist
        
        # Light smoothing
        sdf = gaussian_filter(sdf, sigma=0.3)
        
        return sdf
    
    def extract_cleaned_contours(self, phi_pupil, phi_iris, threshold=0.0):
        """
        Extract and CLEAN contours with morphological operations
        """
        # Extract masks
        pupil_mask = (phi_pupil <= threshold).astype(np.uint8)
        iris_outer_mask = (phi_iris <= threshold).astype(np.uint8)
        
        # Clean pupil mask
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_CLOSE, kernel_small)
        pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Clean iris mask
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        iris_outer_mask = cv2.morphologyEx(iris_outer_mask, cv2.MORPH_CLOSE, kernel_large)
        iris_outer_mask = cv2.morphologyEx(iris_outer_mask, cv2.MORPH_OPEN, kernel_large)
        
        # Create iris region (between pupil and outer boundary)
        iris_mask = iris_outer_mask - pupil_mask
        iris_mask = np.clip(iris_mask, 0, 1).astype(np.uint8)
        
        # Additional cleaning for iris mask
        iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_CLOSE, kernel_small)
        
        # Find contours
        pupil_contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        iris_contours, _ = cv2.findContours(iris_outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (remove small noise)
        if pupil_contours:
            pupil_contours = [c for c in pupil_contours if cv2.contourArea(c) > 50]
        if iris_contours:
            iris_contours = [c for c in iris_contours if cv2.contourArea(c) > 200]
        
        return pupil_mask, iris_mask, pupil_contours, iris_contours
    
    def segment_iris_fixed(self, image):
        """
        MAIN FIXED segmentation function
        """
        # Enhanced preprocessing
        processed_img = self.enhanced_preprocessing(image)
        print("Enhanced preprocessing completed")
        
        # Smart detection with constraints
        center, pupil_radius, iris_radius = self.smart_pupil_iris_detection(image)
        print(f"Detected center: {center}, pupil_radius: {pupil_radius}, iris_radius: {iris_radius}")
        
        # Initialize with constraints
        phi_pupil, phi_iris = self.initialize_constrained_level_sets(
            processed_img.shape, center, pupil_radius, iris_radius
        )
        print("Constrained level sets initialized")
        
        # Evolve with constraints
        final_phi_pupil, final_phi_iris = self.evolve_constrained_contours(
            processed_img, phi_pupil, phi_iris, center, pupil_radius, iris_radius
        )
        
        # Extract cleaned contours
        pupil_mask, iris_mask, pupil_contours, iris_contours = self.extract_cleaned_contours(
            final_phi_pupil, final_phi_iris
        )
        print("Contour extraction completed")
        
        return {
            'original_image': image,
            'processed_image': processed_img,
            'pupil_mask': pupil_mask * 255,
            'iris_mask': iris_mask * 255,
            'combined_mask': (pupil_mask + iris_mask) * 255,
            'pupil_contours': pupil_contours,
            'iris_contours': iris_contours,
            'center': center,
            'pupil_radius': pupil_radius,
            'iris_radius': iris_radius
        }

def visualize_fixed_results(results, save_path="fixed_segmentation_results.png"):
    """
    Enhanced visualization - saves the result to a file instead of showing it
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(results['original_image'], cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Processed image
    axes[0, 1].imshow(results['processed_image'], cmap='gray')
    axes[0, 1].set_title('Preprocessed Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Combined mask
    axes[0, 2].imshow(results['combined_mask'], cmap='gray')
    axes[0, 2].set_title('Combined Segmentation Mask', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Pupil mask
    axes[1, 0].imshow(results['pupil_mask'], cmap='gray')
    axes[1, 0].set_title('Pupil Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Iris mask
    axes[1, 1].imshow(results['iris_mask'], cmap='gray')
    axes[1, 1].set_title('Iris Region Mask', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Overlay
    axes[1, 2].imshow(results['original_image'], cmap='gray')
    
    # Draw contours
    for contour in results['pupil_contours']:
        if len(contour) > 0:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and len(contour) > 3:
                axes[1, 2].plot(contour[:, 0], contour[:, 1], 'b-', linewidth=3, label='Pupil')
    
    for contour in results['iris_contours']:
        if len(contour) > 0:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and len(contour) > 3:
                axes[1, 2].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, label='Iris')
    
    # Mark center
    center = results['center']
    axes[1, 2].plot(center[0], center[1], 'go', markersize=10, label='Center')
    
    axes[1, 2].set_title('FIXED Segmentation Overlay', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Visualization saved at: {save_path}")


def main_fixed():
    """
    Main function with FIXED parameters
    """
    # Update with your image path
    image_path = "/home/prayansh-chhablani/biometrics/iris.jpg"
    
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image loaded successfully. Size: {image.shape}")
    
    # Initialize FIXED GAC with conservative parameters
    fixed_gac = FixedIrisGACSegmentation(
        alpha=0.4,      # Higher curvature weight for smoothness
        beta=0.6,       # Higher edge attraction  
        gamma=0.02,     # MUCH LOWER balloon force
        iterations=250, # Moderate iterations
        dt=0.05,        # Smaller time step
        sigma=1.2       # Moderate smoothing
    )
    
    print("Starting FIXED iris segmentation...")
    results = fixed_gac.segment_iris_fixed(image)
    
    # Visualize results
    visualize_fixed_results(results)
    
    # Save results
    cv2.imwrite('fixed_pupil_mask.png', results['pupil_mask'])
    cv2.imwrite('fixed_iris_mask.png', results['iris_mask'])
    cv2.imwrite('fixed_combined_mask.png', results['combined_mask'])
    
    print(f"\n=== FIXED Segmentation Statistics ===")
    print(f"Image size: {image.shape}")
    print(f"Detected center: {results['center']}")
    print(f"Pupil radius: {results['pupil_radius']}")
    print(f"Iris radius: {results['iris_radius']}")
    print(f"Pupil contours found: {len(results['pupil_contours'])}")
    print(f"Iris contours found: {len(results['iris_contours'])}")
    
    print("FIXED iris segmentation completed successfully!")

if __name__ == "__main__":
    main_fixed()