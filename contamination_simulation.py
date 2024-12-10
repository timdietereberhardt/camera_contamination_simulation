


#### Import needed Libs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from PIL import Image
from scipy.special import binom

import torch
import torch.nn as nn
import torch.nn.functional as F

from noise import pnoise2 


######################################################################################################### Model
# Define the new model structure with latent noise
class ShapeGeneratorModel(nn.Module):
    def __init__(self, noise_dim=100):
        super(ShapeGeneratorModel, self).__init__()
        self.noise_dim = noise_dim
        
        # Linear layers, input dimension includes noise + structured input (x, y, size)
        self.fc1 = nn.Linear(3 + noise_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        
        # Output size to match input for ConvTranspose layers
        self.fc3 = nn.Linear(512, 16 * 16 * 64)
        
        # Transposed convolution layers to upsample to 64x64
        self.conv_transpose1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, xy_size):
        # Generate random noise
        batch_size = xy_size.size(0)
        z = torch.randn(batch_size, self.noise_dim, device=xy_size.device)
        
        # Concatenate structured input (x, y, size) with noise
        x = torch.cat([xy_size, z], dim=1)
        
        # Pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        # Reshape to fit into ConvTranspose layers
        x = x.view(-1, 64, 16, 16)  # Reshape to (batch_size, channels, height, width)
        
        # Upsample to 64x64 using ConvTranspose layers
        x = self.relu(self.conv_transpose1(x))  # 32x32
        x = self.relu(self.conv_transpose2(x))  # 64x64
        x = self.sigmoid(self.conv_transpose3(x))  # Final binary mask (64x64)
        
        return x







######################################################################################################### Total
class OpaqueContaminationGenerator:
    def __init__(self, model_path):
        # Check if a GPU is available and set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Initialize and load the model
        self.model = ShapeGeneratorModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode



    # Update the generate_mask function to work with the new model
    def generate_mask(self, x, y, size, blur_kernel_size, blur_sigma):
        """
        Generates a mask based on the input x, y, size values and applies Gaussian Blur.
        
        :param x: Normalized x-coordinate for mask generation.
        :param y: Normalized y-coordinate for mask generation.
        :param size: Normalized size of the mask.
        :param blur_kernel_size: The size of the Gaussian kernel. Must be odd (default is (15, 15)).
        :param blur_sigma: Standard deviation for Gaussian kernel. If 0, it is computed automatically.
        :return: Tuple containing the original 64x64 mask and the resized 960x960 mask with blur applied.
        """
        # Convert the input into a tensor and move to the correct device (GPU/CPU)
        input_tensor = torch.tensor([[x, y, size]], dtype=torch.float32).to(self.device)
        
        # Generate the mask with the new model
        with torch.no_grad():
            output_mask = self.model(input_tensor)
        
        # Remove batch and channel dimensions
        output_mask = output_mask[0, 0].cpu().numpy()  # Convert to numpy array (64x64)
        
        # Resize the mask to 960x960 using PyTorch's interpolate function
        resized_mask = F.interpolate(torch.tensor(output_mask).unsqueeze(0).unsqueeze(0), size=(960, 960), mode='bilinear', align_corners=False)
        resized_mask = resized_mask.squeeze(0).squeeze(0).numpy()  # Convert back to numpy array
        
        # Apply Gaussian Blur to the resized mask
        blurred_mask = cv2.GaussianBlur(resized_mask, blur_kernel_size, blur_sigma)
        
        return output_mask, blurred_mask


        


    def apply_mask_to_image(self, img_array, x, y, size, mask_color=(0, 0, 0), blur_kernel_size=(15, 15), blur_sigma=0):
        """
        Applies the generated mask to the given image array, with an option to specify the mask color.

        :param img_array: Input image array (960x960 or any size, will be resized)
        :param x: Normalized x-coordinate for mask generation
        :param y: Normalized y-coordinate for mask generation
        :param size: Normalized size of the mask
        :param mask_color: Color of the mask (default is black). Should be a tuple (R, G, B).
        :param blur_kernel_size: The size of the Gaussian kernel. Must be odd (default is (15, 15)).
        :param blur_sigma: Standard deviation for Gaussian kernel. If 0, it is computed automatically.
        :return: Contaminated image with the mask applied
        """
        # Generate the mask
        _, resized_mask = self.generate_mask(x, y, size, blur_kernel_size, blur_sigma)
        
        # Ensure the input image is also 960x960 for applying the mask
        img_resized = cv2.resize(img_array, (960, 960), interpolation=cv2.INTER_LINEAR)
        
        # Normalize mask to range [0, 1]
        resized_mask_normalized = resized_mask.astype(np.float32)

        # If the input image is grayscale, convert it to 3-channel RGB
        if len(img_resized.shape) == 2 or img_resized.shape[2] == 1:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

        # Create a mask color array, matching the shape of the image
        mask_color_array = np.ones_like(img_resized, dtype=np.float32)
        mask_color_array[:, :] = mask_color  # Set the mask color

        # Apply the mask: blend the image with the mask color using the opacity from the resized mask
        contaminated_image = img_resized * (1 - resized_mask_normalized[:, :, np.newaxis]) + \
                            mask_color_array * resized_mask_normalized[:, :, np.newaxis]

        # Ensure the values stay in the range [0, 255] and convert to uint8
        contaminated_image = np.clip(contaminated_image, 0, 255).astype(np.uint8)

        return contaminated_image, resized_mask_normalized


    def TESTapply_mask_to_image_twoc(self, img_array, x, y, size, color_start=(0, 0, 0), color_end=(181, 101, 29), blur_kernel_size=(15, 15), blur_sigma=0):
        """
            Applies the generated mask to the given image array using a gradient between two RGB colors.
        
        :param img_array: Input image array (960x960 or any size, will be resized)
        :param x: Normalized x-coordinate for mask generation
        :param y: Normalized y-coordinate for mask generation
        :param size: Normalized size of the mask
        :param color_start: Start color for the mask gradient (RGB tuple)
        :param color_end: End color for the mask gradient (RGB tuple)
        :param blur_kernel_size: The size of the Gaussian kernel. Must be odd (default is (15, 15)).
        :param blur_sigma: Standard deviation for Gaussian kernel. If 0, it is computed automatically.
        :return: Contaminated image with the gradient mask applied
        """
        # Generate the mask
        _, resized_mask = self.generate_mask(x, y, size, blur_kernel_size, blur_sigma)
        
        # Ensure the input image is 960x960
        img_resized = cv2.resize(img_array, (960, 960), interpolation=cv2.INTER_LINEAR)
        
        # Normalize mask to range [0, 1]
        resized_mask_normalized = resized_mask.astype(np.float32)
        
        # If the input image is grayscale, convert it to 3-channel RGB
        if len(img_resized.shape) == 2 or img_resized.shape[2] == 1:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        # Create a mask color array using linear interpolation between color_start and color_end
        mask_color_array = np.zeros_like(img_resized, dtype=np.float32)
        
        for i in range(3):  # Iterate through the RGB channels
            mask_color_array[:, :, i] = color_start[i] + (color_end[i] - color_start[i]) * resized_mask_normalized

        # Apply the mask: blend the image with the interpolated mask color using the opacity from the resized mask
        contaminated_image = img_resized * (1 - resized_mask_normalized[:, :, np.newaxis]) + \
                            mask_color_array * resized_mask_normalized[:, :, np.newaxis]
        
        # Ensure the values stay in the range [0, 255] and convert to uint8
        contaminated_image = np.clip(contaminated_image, 0, 255).astype(np.uint8)
        
        return contaminated_image, resized_mask_normalized



    def apply_mask_to_image_twoc(self, img_array, x, y, size, color_start=(0, 0, 0), color_end=(181, 101, 29), blur_kernel_size=(15, 15), blur_sigma=0, gamma=0.5):
        """
        Applies the generated mask to the given image array using a gradient between two RGB colors, 
        with a non-linear adjustment to bias towards the first color.
        
        :param img_array: Input image array (960x960 or any size, will be resized)
        :param x: Normalized x-coordinate for mask generation
        :param y: Normalized y-coordinate for mask generation
        :param size: Normalized size of the mask
        :param color_start: Start color for the mask gradient (RGB tuple)
        :param color_end: End color for the mask gradient (RGB tuple)
        :param blur_kernel_size: The size of the Gaussian kernel. Must be odd (default is (15, 15)).
        :param blur_sigma: Standard deviation for Gaussian kernel. If 0, it is computed automatically.
        :param gamma: Exponent for non-linear adjustment of the mask (default is 0.5).
                    Values < 1 bias towards the first color, values > 1 bias towards the second color.
        :return: Contaminated image with the gradient mask applied
        """
        # Generate the mask
        _, resized_mask = self.generate_mask(x, y, size, blur_kernel_size, blur_sigma)
        
        # Ensure the input image is 960x960
        img_resized = cv2.resize(img_array, (960, 960), interpolation=cv2.INTER_LINEAR)
        
        # Normalize mask to range [0, 1]
        resized_mask_normalized = resized_mask.astype(np.float32)
        
        # Apply non-linear transformation to bias towards color_start
        color_weights = resized_mask_normalized ** gamma
        
        # If the input image is grayscale, convert it to 3-channel RGB
        if len(img_resized.shape) == 2 or img_resized.shape[2] == 1:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        # Create a mask color array using linear interpolation between color_start and color_end
        mask_color_array = np.zeros_like(img_resized, dtype=np.float32)
        
        for i in range(3):  # Iterate through the RGB channels
            mask_color_array[:, :, i] = color_start[i] + (color_end[i] - color_start[i]) * color_weights

        # Apply the mask: blend the image with the interpolated mask color using the opacity from the resized mask
        contaminated_image = img_resized * (1 - resized_mask_normalized[:, :, np.newaxis]) + \
                            mask_color_array * resized_mask_normalized[:, :, np.newaxis]
        
        # Ensure the values stay in the range [0, 255] and convert to uint8
        contaminated_image = np.clip(contaminated_image, 0, 255).astype(np.uint8)
        
        return contaminated_image, resized_mask_normalized




######################################################################################################### Semi
class SemiContaminationGenerator:
    def __init__(self, model_path):
        # Check if a GPU is available and set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Initialize and load the model
        self.model = ShapeGeneratorModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode



    def generate_perlin_noise(self, width, height, scale=10):
        """
        Generates a Perlin noise texture of given width and height.
        
        :param width: Width of the noise texture.
        :param height: Height of the noise texture.
        :param scale: Scale of the Perlin noise (affects the level of detail).
        :return: Generated Perlin noise as a 2D NumPy array.
        """
        perlin_noise = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                perlin_noise[i][j] = pnoise2(i / scale, j / scale, octaves=6, persistence=0.5, lacunarity=2.0)
        
        # Normalize the Perlin noise to range [0, 1]
        perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min())
        return perlin_noise
    


    def generate_mask(self, x, y, size, blur_kernel_size=(15, 15), blur_sigma=0):
        """
        Generates a mask based on the input x, y, size values and applies Gaussian Blur.
        
        :param x: Normalized x-coordinate for mask generation.
        :param y: Normalized y-coordinate for mask generation.
        :param size: Normalized size of the mask.
        :param blur_kernel_size: The size of the Gaussian kernel. Must be odd (default is (15, 15)).
        :param blur_sigma: Standard deviation for Gaussian kernel. If 0, it is computed automatically.
        :return: Tuple containing the original 64x64 mask and the resized 960x960 mask with blur applied.
        """
        # Convert the input into a tensor and move to the correct device (GPU/CPU)
        input_tensor = torch.tensor([[x, y, size]], dtype=torch.float32).to(self.device)
        
        # Generate the mask
        with torch.no_grad():
            output_mask = self.model(input_tensor)
        
        # Remove batch and channel dimensions
        output_mask = output_mask[0, 0].cpu().numpy()  # Convert to numpy array (64x64)
        
        # Resize the mask to 960x960 using OpenCV's resize
        resized_mask = cv2.resize(output_mask, (960, 960), interpolation=cv2.INTER_LINEAR)
        
        # Apply Gaussian Blur to the resized mask
        blurred_mask = cv2.GaussianBlur(resized_mask, blur_kernel_size, blur_sigma)
        
        return output_mask, blurred_mask
    

    ### Not in Operation
    # Goal was to shift the vaues and control the strength
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def shift_perlin_noise(self, perlin_noise, shift_factor=0.5):
        """
        Shifts the Perlin noise values closer to 1 by applying a linear transformation.

        :param perlin_noise: The generated Perlin noise array.
        :param shift_factor: The factor by which to shift the values closer to 1 (default is 0.5).
        :return: Perlin noise shifted closer to 1.
        """
        # Apply a linear shift to the noise values
        shifted_noise = perlin_noise * (1 - shift_factor) + shift_factor

        # Apply sigmoid transformation
        #shifted_noise = self.sigmoid(perlin_noise * shift_factor)
        
        return shifted_noise
    
    ###
    def apply_mask_with_perlin_noise(self, img_array, x, y, size, mask_color=(128, 128, 128), blur_kernel_size=(15, 15), blur_sigma=0, noise_scale=10):
        """
        Applies the generated mask to the given image array, embedding a Perlin noise texture in the mask with a specified color.

        :param img_array: Input image array (960x960 or any size, will be resized)
        :param x: Normalized x-coordinate for mask generation
        :param y: Normalized y-coordinate for mask generation
        :param size: Normalized size of the mask
        :param mask_color: Color of the mask, specified as an (R, G, B) tuple (default is gray: (128, 128, 128)).
        :param noise_scale: Scale of the Perlin noise (affects detail level of the noise)
        :param blur_kernel_size: The size of the Gaussian kernel. Must be odd (default is (15, 15)).
        :param blur_sigma: Standard deviation for Gaussian kernel. If 0, it is computed automatically.
        :return: Contaminated image with the Perlin noise mask applied
        """
        # Generate the mask
        _, resized_mask = self.generate_mask(x, y, size, blur_kernel_size, blur_sigma)

        # Generate Perlin noise texture for the mask
        perlin_noise = self.generate_perlin_noise(960, 960, scale=noise_scale)
        
        # Ensure the input image is also 960x960 for applying the mask
        img_resized = cv2.resize(img_array, (960, 960), interpolation=cv2.INTER_LINEAR)
        
        # Normalize mask to range [0, 1]
        resized_mask_normalized = resized_mask.astype(np.float32)
        
        # Apply Perlin noise to the mask (blending with Perlin noise)
        noise_mask = resized_mask_normalized * perlin_noise


        # If the input image is grayscale, convert it to 3-channel RGB
        if len(img_resized.shape) == 2 or img_resized.shape[2] == 1:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

        # Create a mask color array, matching the shape of the image, and apply the specified mask color
        mask_color_array = np.ones_like(img_resized, dtype=np.float32)
        mask_color_array[:, :, 0] = mask_color[0]  # Red channel
        mask_color_array[:, :, 1] = mask_color[1]  # Green channel
        mask_color_array[:, :, 2] = mask_color[2]  # Blue channel

        # Apply the noise mask: blend the image with the mask color using noise as opacity
        contaminated_image = img_resized * (1 - noise_mask[:, :, np.newaxis]) + \
                            mask_color_array * noise_mask[:, :, np.newaxis]

        # Ensure the values stay in the range [0, 255] and convert to uint8
        contaminated_image = np.clip(contaminated_image, 0, 255).astype(np.uint8)

        return contaminated_image, resized_mask_normalized
    

    







######################################################################################################### DROPLET
class DropletGenerator():
    def __init__(self, img_width, img_height, num_channels, model_path):

        # Check if a GPU is available and set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Initialize and load the model
        self.model = ShapeGeneratorModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode

        # Init Vars
        self.img_width = img_width
        self.img_height = img_height
        self.num_channels = num_channels



    def generate_mask(self, x, y, size, blur_kernel_size, blur_sigma):
        """
        Generates a mask based on the input x, y, size values and applies Gaussian Blur.
        
        :param x: Normalized x-coordinate for mask generation.
        :param y: Normalized y-coordinate for mask generation.
        :param size: Normalized size of the mask.
        :param blur_kernel_size: The size of the Gaussian kernel. Must be odd (default is (15, 15)).
        :param blur_sigma: Standard deviation for Gaussian kernel. If 0, it is computed automatically.
        :return: Tuple containing the original 64x64 mask and the resized 960x960 mask with blur applied.
        """
        # Convert the input into a tensor and move to the correct device (GPU/CPU)
        input_tensor = torch.tensor([[x, y, size]], dtype=torch.float32).to(self.device)
        
        # Generate the mask
        with torch.no_grad():
            output_mask = self.model(input_tensor)
        
        # Remove batch and channel dimensions
        output_mask = output_mask[0, 0].cpu().numpy()  # Convert to numpy array (64x64)
        
        # Resize the mask to 960x960 using OpenCV's resize
        resized_mask = cv2.resize(output_mask, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        
        # Apply Gaussian Blur to the resized mask
        blurred_mask = cv2.GaussianBlur(resized_mask, (blur_kernel_size, blur_kernel_size), blur_sigma)
        
        return blurred_mask
    




    class Segment():
        """BEZIER Curve Calculation"""
        def __init__(self, p1, p2, angle1, angle2, **kw):
            self.p1 = p1; self.p2 = p2
            self.angle1 = angle1; self.angle2 = angle2
            self.numpoints = kw.get("numpoints", 100)
            r = kw.get("r", 0.3)
            d = np.sqrt(np.sum((self.p2-self.p1)**2))
            self.r = r*d
            self.p = np.zeros((4,2))
            self.p[0,:] = self.p1[:]
            self.p[3,:] = self.p2[:]
            self.calc_intermediate_points(self.r)

        def bezier(self, points, num=200):
            bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

            N = len(points)
            t = np.linspace(0, 1, num=num)
            curve = np.zeros((num, 2))
            for i in range(N):
                curve += np.outer(bernstein(N - 1, i, t), points[i])
            return curve

        def calc_intermediate_points(self,r):
            self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                        self.r*np.sin(self.angle1)])
            self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                        self.r*np.sin(self.angle2+np.pi)])
            self.curve = self.bezier(self.p,self.numpoints)


    def get_curve(self, points, **kw):
        """Bezier Curve"""
        segments = []
        for i in range(len(points)-1):
            seg = self.Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
            segments.append(seg)
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve


    def get_bezier_curve(self, a, rad=0.2, edgy=0):
        """ given an array of points *a*, create a curve through
        those points. 
        
        *rad* is a number between 0 and 1 to steer the distance of
            control points.
            
        *edgy* is a parameter which controls how "edgy" the curve is,
            edgy=0 is smoothest."""
        p = np.arctan(edgy)/np.pi+.5
        a = self.ccw_sort(a)
        a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
        d = np.diff(a, axis=0)
        ang = np.arctan2(d[:,1],d[:,0])
        f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
        ang = f(ang)
        ang1 = ang
        ang2 = np.roll(ang,1)
        ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
        ang = np.append(ang, [ang[0]])
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        s, c = self.get_curve(a, r=rad, method="var")
        x,y = c.T
        return x,y, a

    def ccw_sort(self, p):
        """Clockwise Sorting"""
        d = p-np.mean(p,axis=0)
        s = np.arctan2(d[:,0], d[:,1])
        return p[np.argsort(s),:]


    def get_random_points(self, n=5, scale=0.8, mindst=None, rec=0):
        """ create n random points in the unit square, which are *mindst*
        apart, then scale them."""
        mindst = mindst or .7/n
        a = np.random.rand(n,2)
        d = np.sqrt(np.sum(np.diff(self.ccw_sort(a), axis=0), axis=1)**2)
        if np.all(d >= mindst) or rec>=200:
            return a*scale
        else:
            return self.get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)


    #### Functions for Masking
    def sq_calc_bounds_white(self, img):
        """
        Find the by the Barrel Tranformation transformed Image in the transferred image

        Detailed description of the function explaining what it does.

        Parameters:
        img within transformed img with white background

        Returns:
        xmin, xmax, ymin, ymax (all Int's): Coordinated of the transformend img
        """
        height, width = img.shape[:2]
        xmin, xmax, ymin, ymax = 0, 0, 0, 0
        # Find xmin
        for i in range(height):
            if not np.all(img[i, :] == 255):
                xmin = i
                break
        # Find xmax
        for i in range(height-1, -1, -1):
            if not np.all(img[i, :] == 255):
                xmax = i
                break
        # Find ymin
        for j in range(width):
            if not np.all(img[:, j] == 255):
                ymin = j
                break
        # Find ymax
        for j in range(width-1, -1, -1):
            if not np.all(img[:, j] == 255):
                ymax = j
                break
        return xmin, xmax, ymin, ymax


        #### Functions for Masking
    def sq_calc_bounds_black(self, img):
        """
        like sq_calc_bounds_white but with black background
        """
        height, width = img.shape[:2]

        xmin, xmax, ymin, ymax = 0, 0, 0, 0

        # Find xmin
        for i in range(height):
            if not np.all(img[i, :] == 0):
                xmin = i
                break
        # Find xmax
        for i in range(height-1, -1, -1):
            if not np.all(img[i, :] == 0):
                xmax = i
                break
        # Find ymin
        for j in range(width):
            if not np.all(img[:, j] == 0):
                ymin = j
                break
        # Find ymax
        for j in range(width-1, -1, -1):
            if not np.all(img[:, j] == 0):
                ymax = j
                break
        return xmin, xmax, ymin, ymax


    def barrel_distortion(self, img, strength):
        """
        Transform a given img with Barrel Transforation

        Detailed description of the function explaining what it does.

        Input:
        img as np.array

        Parameters:
        center (list with two ints (Int, Int)): Center of the Transformation
        strength (Double): Strength of the transformation

        Returns:
        transformed img as np array
        """

        # Init
        height, width = img.shape[:2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))


        center_x = random.randint(10, width)
        center_y = random.randint(10, height)

        center = (center_x, center_y)



        # From Cartesian to Spherical 
        x = x - center[0]
        y = y - center[1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # Barrel-Transformation
        r_distorted = r + strength * r**3

        # Transform in Cartesian Coordinated
        x_distorted = r_distorted * np.cos(theta) + center[0]
        y_distorted = r_distorted * np.sin(theta) + center[1]

        # Interpolation of the disorted image
        distorted_img = cv2.remap(img, x_distorted.astype(np.float32), y_distorted.astype(np.float32), interpolation=cv2.INTER_LINEAR)
        return distorted_img


    def create_white_background(self, height, width):
        """ Create White Background """
        return np.ones((height, width, 3), dtype=np.uint8) * 255  # Füllen Sie die Matrix mit Weiß (255)


    #### User Functions
    def generate_form(self, rad, edgy, n):
        """
        Generates a form using random points and a Bézier curve.

        This function creates a form by first generating a set of random points and then 
        using those points to generate a Bézier curve. The characteristics of the curve 
        are controlled by the 'rad' and 'edgy' parameters.

        Parameters:
        rad (float): A parameter that influences the curvature or radius of the Bézier curve.
        edgy (float): A parameter that controls the sharpness or edginess of the curve.
        n (int): The number of random points to generate.

        Returns:
        tuple: A tuple containing two lists, x and y coordinates, which represent the Bézier curve.
        """
        #print("Start generating...")

        a = self.get_random_points(n=n, scale=1)
        x,y, _ = self.get_bezier_curve(a,rad=rad, edgy=edgy)
        return x,y


    def set_position_form(self, x, y, x_orgin_trans=None, y_orgin_trans=None, size_scale=None):
        """
        Sets the position and scale of a form on a canvas or image.

        This function adjusts the position and size of a form based on provided or randomized 
        translation and scaling values. If no values are provided for the translation or scaling, 
        the function generates random values to position and scale the form within the boundaries 
        of the canvas or image.

        Parameters:
        x (list or ndarray): The x-coordinates of the form's points.
        y (list or ndarray): The y-coordinates of the form's points.
        x_orgin_trans (int, optional): The x-coordinate translation for positioning the form. 
                                    If None, a random value is generated.
        y_orgin_trans (int, optional): The y-coordinate translation for positioning the form. 
                                    If None, a random value is generated.
        size_scale (int, optional): The scaling factor for resizing the form. 
                                    If None, a random value is generated.

        Returns:
        tuple: A tuple containing the transformed x and y coordinates of the form.
        """
        # No values given --> Random init
        if x_orgin_trans==None or y_orgin_trans==None or size_scale==None:
            print("No user defined positions and scaling, randomize...")
            x_orgin_trans = random.randint(10, 0.5*self.img_width-10)
            y_orgin_trans = random.randint(10, 0.5*self.img_height-10)
            size_scale = random.randint(10, 0.5*min(self.img_width, 0.5*self.img_height))

        # Control
        x = np.abs(x * size_scale + x_orgin_trans)
        y = np.abs(y * size_scale + y_orgin_trans)
        return x,y
        
        



    def deploy_zoomin(self, mask, sq_min_x, sq_max_x, sq_min_y, sq_max_y, sq_width, sq_height,
                    raw_img=None, 
                    b_strength=None,
                    ib_gauss_kernel = None,
                    blurr_roi_gauss_kernel=None,
                    mood='black',
                    test=False):

        """
        Deploys a form onto an image, applying barrel distortion, blurring, and blending effects.

        This function takes an input image and a form defined by its x and y coordinates. It then 
        applies a series of transformations, including barrel distortion, Gaussian blurring, and blending 
        with the original image, to create a composite image. The function also handles various 
        parameters such as the strength of the barrel distortion, the size of the Gaussian kernels for 
        blurring, and whether to perform the operations in 'test' mode, where intermediate results 
        are displayed.

        Parameters:
        raw_img (ndarray): The input image onto which the form will be deployed.
        form_x (list or ndarray): The x-coordinates of the form's points.
        form_y (list or ndarray): The y-coordinates of the form's points.
        barrel_strength_low (float, optional): The lower bound for the barrel distortion strength.
        barrel_strength_high (float, optional): The upper bound for the barrel distortion strength.
        ib_gauss_kernel (int, optional): The kernel size for the Gaussian blur applied to the blended image.
        blurr_roi_gauss_kernel (int, optional): The kernel size for the Gaussian blur applied to the region of interest (ROI).
        mask_blurr_gauss_kernel (int, optional): The kernel size for the Gaussian blur applied to the mask.
        mood (str, optional): The background color mode ('black' or 'white') for the distorted region. Default is 'black'.
        test (bool, optional): If True, intermediate steps are visualized for debugging.

        Returns:
        tuple: A tuple containing the fused image (ndarray) and the ground truth mask (ndarray).
        """

        ##### Step 1: Barrel Transformation
        """
        if b_strength is None and barrel_strength_low is None and barrel_strength_high is None:
            barrel_strength_low = 0.00000001
            barrel_strength_high = 0.0001
            
            # Init Parameters for Transformation
            barrel_strength = random.uniform(barrel_strength_low, barrel_strength_high)
        elif barrel_strength_low is not None and barrel_strength_high is not None:
            # Init Parameters for Transformation
            barrel_strength = random.uniform(barrel_strength_low, barrel_strength_high)

        else:
            barrel_strength = b_strength 
        """
        barrel_strength = b_strength 
        
        pre_roi = raw_img[sq_min_x:sq_max_x, sq_min_y:sq_max_y]




        ############################# TEST #############################
        if test:
            print("Pre_roi:  b_strength=", barrel_strength)
            plt.imshow(pre_roi)
            plt.axis('off')
            plt.axis('off')
            plt.show()
        ############################# TEST #############################

        # Do the transformation
        distorted_roi = self.barrel_distortion(pre_roi, barrel_strength)
        ############################# TEST #############################
        if test:
            plt.imshow(distorted_roi)
            plt.axis('off')
            plt.axis('off')
            plt.show()
        ############################# TEST #############################




        ##### Step 2: Background
        # Give that roi a white background
        # Assumption: Droplet is lighter than the raw_image because of lens effect
        if mood == 'white':
            white_background = self.create_white_background(distorted_roi.shape[0], distorted_roi.shape[1])

            # Get it all together
            distorted_roi[np.all(distorted_roi == [0, 0, 0], axis=-1)] = white_background[np.all(distorted_roi == [0, 0, 0], axis=-1)]
            prexmin, prexmax, preymin, preymax = self.sq_calc_bounds_white(distorted_roi)
            prepre_roi = distorted_roi[prexmin:prexmax, preymin:preymax]
        else:
            background = distorted_roi

            # Get it all together
            distorted_roi[np.all(distorted_roi == [0, 0, 0], axis=-1)] = background[np.all(distorted_roi == [0, 0, 0], axis=-1)]
            prexmin, prexmax, preymin, preymax = self.sq_calc_bounds_black(distorted_roi)
            prepre_roi = distorted_roi[prexmin:prexmax, preymin:preymax]

        if test:
            ############################# TEST #############################
            plt.imshow(prepre_roi)
            plt.axis('off')
            plt.axis('off')
            plt.show()
            ############################# TEST #############################



        ##### Step 3: Resize it of old roi
        # Case func calc_bounds found no bounds:
        # TODO: Not Nice
        if prepre_roi.shape[0] == 0 or prepre_roi.shape[1] == 0:
            print("Problem with resize.. Using only distroted:roi")
            content = distorted_roi
        else:
            content = cv2.resize(prepre_roi, ((sq_max_y-sq_min_y), (sq_max_x-sq_min_x)))
        ############################# TEST #############################
        if test:
            plt.imshow(content)
            plt.axis('off')
            plt.axis('off')
            plt.show()
        ############################# TEST #############################



        ##### Step 5: Apply Gaussian blur to ROI
        #if blurr_roi_gauss_kernel is None:
        #    blurr_roi_gauss_kernel = random.choice([11, 13, 15, 17, 19, 21, 23, 25])

        if blurr_roi_gauss_kernel == 0:
            blurred_content = content
        else:
            blurred_content = cv2.GaussianBlur(content, (blurr_roi_gauss_kernel, blurr_roi_gauss_kernel), 0)  # Adjust kernel size (15, 15) as needed


        ############################# TEST #############################
        if test:
            plt.imshow(blurred_content)
            plt.axis('off')
            plt.show()
        ############################# TEST #############################



        ##### Step 6: Apply on old Image
        # Create I_B and blurr it
        #if ib_gauss_kernel is None:
        #    ib_gauss_kernel = 21

        image_b = raw_img.copy()
        image_b[sq_min_x:sq_max_x, sq_min_y:sq_max_y] = blurred_content


        
        ##### Step 7: Blur
        if ib_gauss_kernel > 0:
            image_b = cv2.GaussianBlur(image_b, (ib_gauss_kernel, ib_gauss_kernel), 0)  # Adjust kernel size (15, 15) as needed
        ############################# TEST #############################
        if test:
            print("Gauss Blurr I_B")
            plt.imshow(image_b)
            plt.axis('off')
            plt.show()
        ############################# TEST #############################


        ###### Step 8: Generate
        fused_image=(mask*image_b[:,:,:]) + ((1-mask)*raw_img[:,:,:])
        #TIMfused_image = (mask*image_b[:,:,:]) + ((1-mask)*raw_img[:,:,:])
        fused_image = fused_image.astype(np.uint8)
        if test:
            ############################# TEST #############################   
            print("End of Simulation")
            plt.imshow(fused_image)
            plt.axis('off')
            plt.show()
            ############################# TEST #############################


        return fused_image
        



    def deploy_zoomout(self, mask, sq_min_x, sq_max_x, sq_min_y, sq_max_y, sq_width, sq_height,
                        raw_img=None, 
                        b_strength=None,
                        ib_gauss_kernel = None,
                        blurr_roi_gauss_kernel=None,
                        mood='black',
                        test=False):

        """
        Deploys a form onto an image, rotating and blurring a selected region.

        This function takes an input image and a form defined by its x and y coordinates. It then 
        applies a series of transformations, including rotation, resizing, Gaussian blurring, 
        and blending with the original image, to create a composite image. The function also handles 
        various parameters such as the size of the Gaussian kernels for blurring and whether to perform 
        the operations in 'test' mode, where intermediate results are displayed.

        Parameters:
        raw_img (ndarray): The input image onto which the form will be deployed.
        form_x (list or ndarray): The x-coordinates of the form's points.
        form_y (list or ndarray): The y-coordinates of the form's points.
        ib_gauss_kernel (int, optional): The kernel size for the Gaussian blur applied to the blended image.
        blurr_roi_gauss_kernel (int, optional): The kernel size for the Gaussian blur applied to the region of interest (ROI).
        mood (str, optional): The background color mode ('black' or 'white') for the distorted region. Default is 'black'.
        test (bool, optional): If True, intermediate steps are visualized for debugging.

        Returns:
        tuple: A tuple containing the fused image (ndarray) and the ground truth mask (ndarray).
        """

        ##### Step 1: Extract Region of Interest (ROI)
        pre_roi = raw_img   #[sq_min_x:sq_max_x, sq_min_y:sq_max_y]
        ############################# TEST #############################
        if test:
            print("Original ROI:")
            plt.imshow(pre_roi)
            plt.axis('off')
            plt.show()
        ############################# TEST #############################



        ##### Step 2: Rotate the ROI by 180 degrees
        rotated_roi = cv2.rotate(pre_roi, cv2.ROTATE_180)
        
        ############################# TEST #############################
        if test:
            print("Rotated ROI:")
            plt.imshow(rotated_roi)
            plt.axis('off')
            plt.show()
        ############################# TEST #############################



        ##### Step 3: Resize it to a smaller version
        miniaturized_roi = cv2.resize(rotated_roi, ((sq_max_y-sq_min_y) // 2, (sq_max_x-sq_min_x) // 2))
        ############################# TEST #############################
        if test:
            print("Miniaturized ROI:")
            plt.imshow(miniaturized_roi)
            plt.axis('off')
            plt.show()
        ############################# TEST #############################



        ##### Step 4: Apply Gaussian blur to the resized ROI
        if blurr_roi_gauss_kernel == 0:
            blurred_content = miniaturized_roi
        else:
            blurred_content = cv2.GaussianBlur(miniaturized_roi, (blurr_roi_gauss_kernel, blurr_roi_gauss_kernel), 0)
        
        ############################# TEST #############################
        if test:
            print("Blurred Miniaturized ROI:")
            plt.imshow(blurred_content)
            plt.axis('off')
            plt.show()
        ############################# TEST #############################



        ##### Step 5: Embed the blurred content back into the original image
        image_b = raw_img.copy()
        # Resize the blurred content to match the region size
        resized_blurred_content = cv2.resize(blurred_content, (sq_max_y - sq_min_y, sq_max_x - sq_min_x))
        image_b[sq_min_x:sq_max_x, sq_min_y:sq_max_y] = resized_blurred_content



        ##### Step 6: Apply Gaussian blur to the whole image
        if ib_gauss_kernel > 0:
            image_b = cv2.GaussianBlur(image_b, (ib_gauss_kernel, ib_gauss_kernel), 0)
        ############################# TEST #############################
        if test:
            print("Final Image:")
            plt.imshow(image_b)
            plt.axis('off')
            plt.show()
        ############################# TEST #############################



        ###### Step 7: Generate the final image by blending with the original using the mask
        fused_image=(mask*image_b[:,:,:]) + ((1-mask)*raw_img[:,:,:])
        #fused_image = fused_image.astype(np.uint32)
        fused_image = fused_image.astype(np.uint8)
        
        ############################# TEST #############################   
        if test:
            print("End of Simulation:")
            plt.imshow(fused_image)
            plt.axis('off')
            plt.show()
        ############################# TEST #############################
        return fused_image

            




    def simulate_droplet(self, f_C, raw_img, x, y, size,
                            mask_blurr_gauss_kernel=None, blur_sigma=0,
                            b_strength=None,
                            ib_gauss_kernel = None,
                            blurr_roi_gauss_kernel=None,
                            mood='black',
                            mask_threshold=0.5,
                            test=False):

        # Create Mask with Model
        mask = self.generate_mask(x, y, size, mask_blurr_gauss_kernel, blur_sigma)




        if raw_img.shape[0] != self.img_width or raw_img.shape[1] != self.img_height:
            raise ValueError("Given shapes are not matching!")
        else:
            # Safe ground Truth
            #mask = np.where(mask >= mask_threshold, 1, 0)
            # Convert mask to uint8
            gt_mask = np.uint8(mask.copy() * 255)  # Convert binary mask to 0-255 scale
            mask = cv2.GaussianBlur(gt_mask, (mask_blurr_gauss_kernel, mask_blurr_gauss_kernel), 0)  # Adjust kernel size (15, 15) as needed
            ############################# TEST #############################
            if test:
                    print("GT-Mask:")
                    plt.imshow(gt_mask)
                    plt.axis('off')
                    plt.colorbar()
                    plt.show()
                    print("Mask:")
                    plt.imshow(mask)
                    plt.axis('off')
                    plt.colorbar()
                    plt.show()
            ############################# TEST #############################
            
            
            # Check if Droplet is to small or noise
            indices = np.where(gt_mask > 0)
            if indices[0].size > 0:
                # indices[0] --> x_coordinates # indices[1] --> y_coordinates
                # Sq Square Data Properties
                sq_min_x = np.min(indices[0])
                sq_min_y = np.min(indices[1])
                sq_max_x = np.max(indices[0])
                sq_max_y = np.max(indices[1])
                sq_width = sq_max_x - sq_min_x
                sq_height = sq_max_y - sq_min_y

                # Create approximated square mask
                sq_mask = np.ones(raw_img.shape[:2], dtype=np.uint8)
                sq_mask[sq_min_x:sq_max_x, sq_min_y:sq_max_y] = 255

                # Reshape mask to have the same number of dimensions as the RGB images
                mask = mask[:, :, np.newaxis]
                mask = mask / np.max(mask)      # Normalize Mask from [0,255] to [0,1]
                ############################# TEST #############################
                if test:
                    print("SQ-Mask:")
                    plt.imshow(sq_mask)
                    plt.axis('off')
                    plt.show()
                ############################# TEST #############################



            





                # OM Classifier
                ####################################################################################################################
                f_D = (0.5*max(sq_max_x,sq_max_y)) / (1.33-1)

                if f_D > f_C:
                    ### ZoomIn
                    fused_image = self.deploy_zoomin(mask, sq_min_x, sq_max_x, sq_min_y, sq_max_y, sq_width, sq_height,
                                    raw_img, 
                                    b_strength,
                                    ib_gauss_kernel,
                                    blurr_roi_gauss_kernel,
                                    mood, test)
                    print("Case: ZoomIn")
                else:
                    ### ZoomOut
                    fused_image = self.deploy_zoomout(mask, sq_min_x, sq_max_x, sq_min_y, sq_max_y, sq_width, sq_height,
                                    raw_img, 
                                    b_strength,
                                    ib_gauss_kernel,
                                    blurr_roi_gauss_kernel,
                                    mood, test)
                    print("Case: ZoomOut")
            

            ####################################################################################################################
            else:
                fused_image = raw_img.copy()
                gt_mask = np.zeros((raw_img.shape[:2] ), dtype=np.uint8)

            print(type(fused_image))

            return fused_image, gt_mask
        

"""
Parameter:

WoodScape Dataset:
fd_x = 432.201
fd_y = 326.177
Mean = 379.189 approx 380

"""




















#-----------------------------------------------------------OtherPaper-----------------------------------------------------------#
"""
Raindrops on Windshield: Dataset and Lightweight Gradient-Based Detection Algorithm

Autonomous vehicles use cameras as one of the
primary sources of information about the environment. Adverse
weather conditions such as raindrops, snow, mud, and others,
can lead to various image artifacts. Such artifacts significantly
degrade the quality and reliability of the obtained visual
data and can lead to accidents if they are not detected in
time. This paper presents ongoing work on a new dataset
for training and assessing vision algorithms’ performance for
different tasks of image artifacts detection on either camera
lens or windshield. At the moment, we present a publicly
available set of images containing 8190 images, of which 3390
contain raindrops. Images are annotated with the binary mask
representing areas with raindrops. We demonstrate the appli-
cability of the dataset in the problems of raindrops presence
detection and raindrop region segmentation. To augment the
data, we also propose an algorithm for data augmentation
which allows the generation of synthetic raindrops on images.
Apart from the dataset, we present a novel gradient-based
algorithm for raindrop presence detection in a video sequence.
The experimental evaluation proves that the algorithm reliably
detects raindrops. Moreover, compared with the state-of-the-art
cross-correlation-based algorithm [1], the proposed algorithm
showed a higher quality of raindrop presence detection and
image processing speed, making it applicable for the self-check
procedure of real autonomous systems. The dataset is available
at github.com/EvoCargo/RaindropsOnW indshield.


https://arxiv.org/abs/2104.05078

https://github.com/Evocargo/RaindropsOnWindshield

"""


def op_make_bezier(xys):
    n = len(xys)
    combinations = op_pascal_row(n-1)
    def bezier(ts):
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier

def op_pascal_row(n, memo={}):
    if n in memo:
        return memo[n]
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n % 2 == 0:
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    memo[n] = result
    return result

def normalize(data):
    # Standardnormalisierung
    mean = np.mean(data)
    std_dev = np.std(data)
    standardized_data = (data - mean) / std_dev
    
    # Min-Max-Skalierung
    min_val = np.min(standardized_data)
    max_val = np.max(standardized_data)
    scaled_data = (standardized_data - min_val) / (max_val - min_val)
    
    return scaled_data

class Raindrop:
    def __init__(self, radius=50, center=(0, 0)):
        self.radius = radius
        self.center = center
    
    def generate_circle_points(self, num_points=100):
        angles = np.linspace(0, 2 * np.pi, num_points)
        x_points = self.center[0] + self.radius * np.cos(angles)
        y_points = self.center[1] + self.radius * np.sin(angles)
        return x_points, y_points
    
    def generate_ellipse_points(self, num_points=100):
        angles = np.linspace(0, 2 * np.pi, num_points)
        x_points = self.center[0] + self.radius * np.cos(angles)
        y_points = self.center[1] + 1.3 * self.radius * np.sin(angles)
        return x_points, y_points
    


    def generate_egg_shape_points(self, num_points=100, A=None, B=None, C=None, D=None, test=None):
        if A is None or B is None or C is None or D is None:
            A = (self.center[0] + 0.5 * self.radius, self.center[1] - self.radius)
            B = (self.center[0] + 0.25 * self.radius, self.center[1] + 0.5 * self.radius)
            C = (self.center[0] - 0.25 * self.radius, self.center[1] + 0.5 * self.radius)
            D = (self.center[0] - 0.5 * self.radius, self.center[1] - self.radius)
        
        ts = np.linspace(0, 1, num_points)
        
        # Erste Kurve
        xys = [A, B, C]
        bezier = op_make_bezier(xys)
        points1 = np.array(bezier(ts))
        
        # Zweite Kurve (umgekehrte Reihenfolge der Kontrollpunkte)
        xys = [C, D, A]  # Beachten Sie die umgekehrte Reihenfolge hier
        bezier = op_make_bezier(xys)
        points2 = np.array(bezier(ts))

        
        
        # Verbinden der beiden Kurven
        points = np.concatenate([points1, points2], axis=0)

                # Normalize
        points[:, 0] = normalize(np.array(points[:, 0]))
        points[:, 1] = normalize(np.array(points[:, 1]))

        ############################# TEST #############################
        # Plot
        if test:
            plt.plot(points[:, 0], points[:, 1], 'g-', label='Egg Shape (Drop)')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Shapes')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()
            plt.grid(True)
            plt.show()
        ############################# TEST #############################



        
        return points[:, 0], points[:, 1]
        
    """
    def deploy(self, raw_img=None, form_x=None, form_y=None, 
                barrel_strength_low=None, barrel_strength_high=None, 
                ib_gauss_kernel = None,
                blurr_roi_gauss_kernel=None,
                mask_blurr_gauss_kernel=None,
                brightness_factor=None,
                mood='black',
                test=False):
    """


    def op_employ(self, raw_img=None, form_x=None, form_y=None, 
                  img_gauss_kernel=None, mask_blurr_gauss_kernel=None,
                  test=None):



        # (1) Create a mask
        mask = np.zeros(raw_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(list(zip(form_x, form_y)), dtype=np.int32)], 255)

        if test:
            ############################# TEST #############################
            plt.imshow(mask)
            plt.show()
            ############################# TEST #############################


        # Blurr Image b
        image_b = raw_img.copy()

        # Apply Gaussian blur to ROI
        #if mask_blurr_gauss_kernel is None:
        #    mask_blurr_gauss_kernel = 21

        image_b = cv2.GaussianBlur(image_b, (mask_blurr_gauss_kernel, mask_blurr_gauss_kernel), 0)  # Adjust kernel size (15, 15) as needed

        if test:
            ############################# TEST #############################
            print("Blurred Image_b")
            plt.imshow(image_b)
            plt.show()
            ############################# TEST #############################





        # (2) Blurr Mask
        ## Blurr the mask and invert it
        #if img_gauss_kernel is None:
        #    img_gauss_kernel = 91


        gt_mask = mask.copy()   # Safe it for ground truth
        mask = cv2.GaussianBlur(mask, (img_gauss_kernel, img_gauss_kernel), 0)  # Adjust kernel size (15, 15) as needed



        if test:
            ############################# TEST #############################
            print("Maske")
            plt.imshow(mask)
            plt.show()
            ############################# TEST #############################



        # (3) Implementation
        # Reshape mask to have the same number of dimensions as the RGB images
        mask = mask[:, :, np.newaxis]

        # Normalize Mask from [0,255] to [0,1]
        mask = mask / np.max(mask)

        # CALC
        fused_image=(mask*image_b[:,:,:]) + ((1-mask)*raw_img[:,:,:])
        #TIMfused_image = (mask*image_b[:,:,:]) + ((1-mask)*raw_img[:,:,:])
        fused_image = fused_image.astype(np.uint8)


        if test:
            ############################# TEST #############################   
            print("End of Simulation")
            plt.imshow(fused_image)
            plt.show()
            print("Mask")
            plt.imshow(gt_mask)
            plt.show()
            ############################# TEST #############################


        return fused_image, gt_mask













