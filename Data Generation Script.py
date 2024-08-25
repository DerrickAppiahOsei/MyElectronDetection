import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py

class ImageProcessor:
    """Image generation class"""    
    
    def __init__(self, 
                num_to_generate,
                tiff_path, 
                dark_noise_path, 
                canvas_size = (64, 64), 
                max_electron_hits = 3):
        """Initializes the ImageProcessor class

        Args:
            num_to_generate (int): Number of images to generate
            tiff_path (str): Location of the tiff file that contains the electron hits
            dark_noise_path (str): Path to the noisy data
            canvas_size (tuple, optional): Size of the canvas to place the images on. Defaults to (64, 64).
            max_electron_hits (int, optional): Maximum number of electron hits to place on the canvas. Defaults to 272.
        """        
        
        self.tiff_path = tiff_path
        self.dark_noise_path = dark_noise_path
        self.canvas_size = canvas_size
        self.num_to_generate = num_to_generate
        self.max_electron_hits = max_electron_hits
        
        # Loads the images from the tiff file
        self.images = self.load_images_from_tiff(tiff_path)
        self.dark_noise_images = self.load_images_from_tiff(dark_noise_path)
        
        # Creates a dark stack of the same size as the canvas
        self.dark_noise = self.dark_stack(self.canvas_size[0])
    
    def load_images_from_tiff(self, tiff_path):
        """Loads the images from a tiff file

        Args:
            tiff_path (str): Path to the tiff file

        Returns:
            list: List of images
        """        
        with Image.open(tiff_path) as img:
            images = []
            for i in range(img.n_frames):
                img.seek(i)
                images.append(np.array(img))
            return images
        
    def noisy(self, noise_typ, image): 
        """Adds noise to the images
        
        Args:
            noise_typ (str): Type of noise to add
            image (numpy array): Image to add noise to
        
        Returns:
            numpy array: Noisy image
        """
        if noise_typ == "gauss":
            row, col = image.shape
            mean = 0
            var = 0.0001
            sigma = var**0.5
            threshold = 8
            gauss = np.random.normal(mean, sigma, (row, col))
            tnoisy = image + gauss

            tnoisy[tnoisy < threshold] = 0
            noisy = np.round(tnoisy)
            return noisy

    def deadcorr(self, image):
        """Corrects the dead pixel within the dark reference frame by interpolating from near positions.
        
        Args:
            image (numpy array): Image to correct
        """        
        temp = image.copy()
        temp[:, 248] = 0.5 * temp[:, 247] + 0.5 * temp[:, 246]
        return temp

    def dark_stack(self, imgsize):
        """Creates a dark stack of the same size as the canvas.
        
        Args:
            imgsize (int): Size of the images in the stack
        """        
        dark_noise_stack_corrected = [self.deadcorr(image) for image in self.dark_noise_images]
        dark_noise_stack_cropped = [image[512:512+imgsize, 512:512+imgsize] for image in dark_noise_stack_corrected]
        return dark_noise_stack_cropped

    # def place_image_on_canvas(self, positions, box_size=2):
    #     """Places the electron hits on the canvas.
        
    #     Args:
    #         positions (int): Number of electron hits to place on the canvas
    #     """        
    #     canvas = np.zeros(self.canvas_size, dtype=np.uint8)
    #     height, width = self.images[0].shape
    #     max_x = self.canvas_size[1]
    #     max_y = self.canvas_size[0]
    #     bounding_boxes = []
    #     centers = []
    #     index_ = []
    #     bounding_boxes_training = np.zeros((self.max_electron_hits, 5))
    #     centers_training = np.zeros((self.max_electron_hits, 3))
        
    #     for i in range(positions):
    #         x = random.randint(1 - width//2, max_x - width//2 - 1)
    #         y = random.randint(1 - height//2, max_y - height//2 - 1)
    #         index = random.randint(0, len(self.images) - 1)
    #         hit = self.images[index]

    #         y_min = y
    #         y_max = y + height
    #         x_min = x 
    #         x_max = x + width
            
    #         x_center = x + width / 2 
    #         y_center = y + height / 2
            
    #         if y_min < 0:
    #             hit = hit[y_min*-1:, :]
    #             y_min = 0

    #         if y_max > max_y:
    #             hit = hit[:-(y_max-max_y), :]
    #             y_max = max_y
                
    #         if x_min < 0:
    #             hit = hit[:, x_min*-1:]
    #             x_min = 0
            
    #         if x_max > max_x:
    #             hit = hit[:, :-(x_max-max_x)]
    #             x_max = max_x
            
    #         canvas[y_min:y_max, x_min:x_max] = hit
    #         bounding_boxes.append((x_center-1, y_center-1, x_center+1, y_center+1))
    #         bounding_boxes_training[i, 1:] = [x_center-1, y_center-1, x_center+1, y_center+1]
    #         centers.append((x_center, y_center))
    #         centers_training[i, 1:] = [x_center, y_center]
    #         index_.append(index)
    def place_image_on_canvas(self, positions=3, box_size=2):
        """Places the electron hits on the canvas.
        
        Args:
            positions (int): Number of electron hits to place on the canvas
        """        
        canvas = np.zeros(self.canvas_size, dtype=np.uint8)
        height, width = self.images[0].shape
        max_x = self.canvas_size[1]
        max_y = self.canvas_size[0]
        bounding_boxes = []
        centers = []
        index_ = []
        bounding_boxes_training = np.zeros((self.max_electron_hits, 5), dtype=np.float32)
        centers_training = np.zeros((self.max_electron_hits, 3), dtype=np.float32)
        
        for i in range(positions):
            # Ensure the top-left corner is within bounds
            x = random.randint(0, max_x - width)
            y = random.randint(0, max_y - height)
            index = random.randint(0, len(self.images) - 1)
            hit = self.images[index]

            y_min = y
            y_max = y + height
            x_min = x 
            x_max = x + width
            
            # Calculate the exact center based on pixel intensity
            total_intensity = np.sum(hit)
            if total_intensity > 0:
                y_coords, x_coords = np.indices(hit.shape)
                x_center = x + np.sum(x_coords * hit) / total_intensity
                y_center = y + np.sum(y_coords * hit) / total_intensity
            else:
                x_center = x + width / 2
                y_center = y + height / 2
            
            canvas[y_min:y_max, x_min:x_max] = hit
            bounding_boxes.append((x_center-1, y_center-1, x_center+1, y_center+1))
            bounding_boxes_training[i, 0] = 1
            bounding_boxes_training[i, 1:] = [x_center-1, y_center-1, x_center+1, y_center+1]
            centers.append((x_center, y_center))
            centers_training[i, 0] = 1
            centers_training[i, 1:] = [x_center, y_center]
            index_.append(index)
        
        canvas = self.noisy('gauss', canvas)
        noise_int = np.random.randint(len(self.dark_noise))
        canvas = canvas + self.dark_noise[noise_int]
            
        return (canvas, bounding_boxes, bounding_boxes_training, centers, centers_training, index_, positions, noise_int)


    def generate_multiple_images(self):
        """Generates multiple images"""        
        results = []
        for i in tqdm(range(self.num_to_generate), desc="Generating images"):
            positions = random.randint(1, self.max_electron_hits)
            results.append(self.place_image_on_canvas(positions))
        return results

    def save_to_h5(self, data, filename):
        """Saves data to an HDF5 file
        
        Args:
            data (list): List of data to save
            filename (str): Path to the HDF5 file
        """
          # with h5py.File(filename, 'w') as h5_file:
        #     for i, item in enumerate(data):
        #         h5_file.create_dataset(f'image_{i}_image', data=item[0])
        #         # h5_file.create_dataset(f'image_{i}_bounding_boxes', data=np.array(item[1]))
        #         # h5_file.create_dataset(f'image_{i}_bounding_boxes_training', data=item[2])
        #         # h5_file.create_dataset(f'image_{i}_center_positions', data=np.array(item[3]))
        #         h5_file.create_dataset(f'image_{i}_center_positions_training', data=item[4])
        #         # h5_file.create_dataset(f'image_{i}_index', data=np.array(item[5]))
        #         # h5_file.create_dataset(f'image_{i}_position', data=item[6])
        #         # h5_file.create_dataset(f'image_{i}_noise', data=item[7])
        with h5py.File(filename, 'w') as h5_file:
            theimages = h5_file.create_dataset('images',shape = (self.num_to_generate, 64, 64), dtype = 'uint8')
            thecenters = h5_file.create_dataset('centers_training', shape = (self.num_to_generate, 3, 3), dtype = 'float32')
            for i, item in enumerate(data):
                theimages[i] = item[0]
                thecenters[i] = item[4]
tiff_path = 'DerrickAppiahOsei/MyElectronDetection/Raw Camera Data/200kV_98000electron.tif'
dark_noise_path = 'DerrickAppiahOsei/MyElectronDetection/Raw Camera Data/1000fps_fullRolling.tif'

processor = ImageProcessor(100000, tiff_path, dark_noise_path, max_electron_hits=3)
data = processor.generate_multiple_images()
processor.save_to_h5(data, 'DerrickAppiahOsei/MyElectronDetection/Data Generated/100Kpadded.h5')
