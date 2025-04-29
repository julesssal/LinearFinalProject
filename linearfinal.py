# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import cv2


#function to create missing pixels
def create_missing_pixels(img, missing_rate=0.2):
    """Randomly remove a percentage of the image's pixels."""
    corrupted_img = img.copy()
    mask = np.random.rand(*img.shape) < missing_rate  # Create random mask
    corrupted_img[mask] = np.nan  # Set missing pixels to NaN
    return corrupted_img, mask

def get_neighbors(img, i, j, window_size=1):
    """Extract neighboring pixel values around (i, j) with NaN handling."""
    neighbors = []
    rows, cols = img.shape
    for di in range(-window_size, window_size + 1):
        for dj in range(-window_size, window_size + 1):
            if di == 0 and dj == 0:
                continue  # Skip the center pixel
            ni, nj = i + di, j + dj  # Neighbor coordinates
            if 0 <= ni < rows and 0 <= nj < cols:
                value = img[ni, nj]
                if np.isnan(value):
                    value = 0  # Replace missing neighbor with 0
                neighbors.append(value)
            else:
                neighbors.append(0)  # Border treated as 0
    return neighbors

def psnr(mse):
    """Calculate Peak Signal to Noise Ratio from Mean Squared Error."""
    return 10 * np.log10(1 / mse)

# --- Load Real Image ---

# Path to image
img_path = 'C:/Users/JSale/LinearFinal/facec.jpg'  
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  #make image grayscale
if img is None:
    raise ValueError("Image not found!")

# Resize image to manageable size
img = cv2.resize(img, (100, 100))  # 100x100 pixels

# Normalize pixel values to range [0,1]
original_img = img / 255.0

#Create the missing pixels

missing_rate = 0.2  # .x = percentage missing, .2-= 20%
corrupted_img, missing_mask = create_missing_pixels(original_img, missing_rate) #create image with the pixels gone

#Prep Training Data

X_train = []  # List for neighbor feature vectors
y_train = []  # List for target center pixel values

rows, cols = original_img.shape
window_size = 2  # 5x5 neighbor window (excluding center)

for i in range(window_size, rows - window_size):
    for j in range(window_size, cols - window_size):
        if not np.isnan(corrupted_img[i, j]):  # Center pixel must be known
            neighbors = get_neighbors(corrupted_img, i, j, window_size)
            X_train.append(neighbors)  #creates a matrix where each row is a vector of neighbors for a known pixel.
            y_train.append(corrupted_img[i, j])  # vector where each entry is the true center pixel value.
            #In matrix notation: 𝑋 ∈ 𝑅^m×k and 𝑦 ∈ 𝑅^m where: 
            # m = number of known center pixels you collected
            # k = number of neighbors per center (for 5x5 window, excluding center, k=24)

X_train = np.array(X_train)  # Convert to numpy array
y_train = np.array(y_train)

# Train Rregression Modle

model = Ridge(alpha=.10)  # Regularized linear regression β=(X^TX+αI)^−1 * X^Ty
#model = LinearRegression() #Linear Regression β=(X^TX)^−1 * X^Ty
# solving a system of linear equations to find the best weights for the neighbors
model.fit(X_train, y_train)  # Fit model to training data

#Pass over image multiple times to reconstruct

reconstructed_img = corrupted_img.copy()  # Start with corrupted image

# Perform multiple passes to fill more missing pixels each time
for pass_num in range(1):  # 10 prediction passes
    for i in range(window_size, rows - window_size):
        for j in range(window_size, cols - window_size):
            if np.isnan(reconstructed_img[i, j]):  # Only predict missing pixels
                neighbors = get_neighbors(reconstructed_img, i, j, window_size)
                pred = model.predict(np.array(neighbors).reshape(1, -1))
                reconstructed_img[i, j] = pred.item()  # Insert predicted value

#Performance Evaluation

# Only evaluate on pixels that were missing and now reconstructed
valid_pixels = ~np.isnan(reconstructed_img[missing_mask])

# Calculate MSE (Mean Squared Error)
mse = mean_squared_error(
    original_img[missing_mask][valid_pixels],
    reconstructed_img[missing_mask][valid_pixels]
)

# Calculate PSNR (Peak Signal to Noise Ratio)
psnr_value = psnr(mse)

print(f"MSE: {mse:.6f}")
print(f"PSNR: {psnr_value:.2f} dB")

#Create Plot of the Results

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(np.where(missing_mask, 1, corrupted_img), cmap='gray')
axes[1].set_title('Corrupted Image (Missing 20%)')
axes[1].axis('off')

axes[2].imshow(reconstructed_img, cmap='gray')
axes[2].set_title('Reconstructed Image (Regression)')
axes[2].axis('off')

# Overall title showing performance
#plt.suptitle(f"MSE: {mse:.6f} | PSNR: {psnr_value:.2f} dB", fontsize=16)
plt.tight_layout()
plt.show()
#pseudoinverse 