'''
This algorithm will transform one image into another image in a smooth animation.
It is assumed that the object of interest will be placed upon a white background 
in both the input and output images.

Find the Pixel Paths
step 1: classify each pixel in both images as colored or not colored
step 2: count colored pixels in both images
step 3: get a random assignment of pixels from image 1 to image 2. 
        - Think if it as a function 
        where the input is a pixel from image 1 and the output is a pixel from image 2.
        The function will be one to many or many to one, based on the relative number 
        of colored pixels between the images.
step 4: Generate n variants of the initial function. This consists of randomly swapping output 
        pixels for a given input pixel, or vice versa.
step 5: Score the original image and all the generated variants based on the aggregate cartesian 
        distances of all the function inputs and output. For example, if pixel (223,400) maps to 
        pixel (47,388), then the cartesian distance would be ((223-47)^2+(400-388)^2)^.5.
        The new image will be the one with the smallest aggregate distance.
step 6: Iterate steps 4 and 5 until a convergence is reached, for example, until n generations
        occur without an improvement.

Create the Animation
step 1: Determine the number of frames in the desired animation based on duration and FPS.
step 2: Determine the RGB color of the beginning and the end of each pixel.
step 3: Generate each frame of the animation by placing each pixel at a point between the 
        start location and the end location, and choose the color of the pixel by using 
        the current frame number divided by the total number of frames as a weight in the 
        weighted average of the beginning and end color. In this way, the color will change 
        smoothly from the beginning to the end of the animation.
'''

import numpy as np
from PIL import Image, ImageDraw
import random
import copy
import subprocess
import os

np.set_printoptions(suppress=True, precision=2, linewidth=140)

class Transformer:
    def __init__(self,image1,image2):
        self.im1_path = image1
        self.im2_path = image2
        self.white_threshold = 224
        self.population_size = 5
        self.mutation_rate = .01
        self.convergence_threshold = 200
        self.converged = False
        self.same_score_count = 0
        self.generation_count = 0
        self.number_of_frames = 60
        self.frames_per_second = 60
        self.run()
    def run(self):
        print("Opening images")
        self.open_images()
        print("Classifying pixels")
        self.classify_pixels()
        print("Randomly initializing transition function")
        self.get_random_pixel_mappings()
        print("Performing iterative algorithm")
        while not self.converged:
            self.mutate()
        print("\n    "+str(self.generation_count) + " generations, converged!")
        self.generate_frames()
        print("Rendering video")
        self.render_video()
    def open_images(self):
        self.im1 = Image.open(self.im1_path)
        self.np_im1 = np.asarray(self.im1)
        self.im2 = Image.open(self.im2_path)
        self.np_im2 = np.asarray(self.im2)
        assert(self.im1.size==self.im2.size)
        print("    Image shape: " + str(self.np_im1.shape))
    def classify_pixels(self):
        im1_means = np.mean(self.np_im1,axis=2)
        im2_means = np.mean(self.np_im2,axis=2)
        self.im1_wherecolored = im1_means < self.white_threshold
        self.im2_wherecolored = im2_means < self.white_threshold
        self.im1_colored_pixel_count = np.sum(self.im1_wherecolored)
        self.im2_colored_pixel_count = np.sum(self.im2_wherecolored)
        if self.im1_colored_pixel_count == self.im2_colored_pixel_count:
            self.relationship = "one to one"
        elif self.im1_colored_pixel_count > self.im2_colored_pixel_count:
            self.relationship = "many to one"
        else:
            self.relationship = "one to many"
        print("    Relationship: " + str(self.relationship))
        print("    Pixel mappings: " + str(max(self.im1_colored_pixel_count,self.im2_colored_pixel_count)))
    def get_random_pixel_mappings(self):
        in_pixels = np.argwhere(self.im1_wherecolored)
        in_pixels = [tuple(item) for item in np.argwhere(self.im1_wherecolored)]
        out_pixels= np.argwhere(self.im2_wherecolored)
        out_pixels = [tuple(item) for item in np.argwhere(self.im2_wherecolored)]
        self.pairings = []
        if self.relationship == "one to one":
            for i in range(len(in_pixels)):
                in_pix = in_pixels.pop(i)
                out_pix_index = random.randint(0,len(out_pixels)-1)
                out_pix = out_pixels.pop(out_pix_index)
                self.pairings.append([in_pix,out_pix])
        elif self.relationship == "many to one":
            higher_number = self.im1_colored_pixel_count//self.im2_colored_pixel_count+1
            higher_number_count = self.im1_colored_pixel_count%self.im2_colored_pixel_count
            higher_numbers_placed = 0
            for i in range(len(out_pixels)):
                out_pix = out_pixels.pop(0)
                if higher_numbers_placed < higher_number_count:
                    for j in range(higher_number):
                        in_pix_index = random.randint(0,len(in_pixels)-1)
                        in_pix = in_pixels.pop(in_pix_index)
                        self.pairings.append([in_pix,out_pix])
                    higher_numbers_placed += 1
                else:
                    for j in range(higher_number-1):
                        in_pix_index = random.randint(0,len(in_pixels)-1)
                        in_pix = in_pixels.pop(in_pix_index)
                        self.pairings.append([in_pix,out_pix])
        else: # self.relationship == "one to many":
            higher_number = self.im2_colored_pixel_count//self.im1_colored_pixel_count+1
            higher_number_count = self.im2_colored_pixel_count%self.im1_colored_pixel_count
            higher_numbers_placed = 0
            for i in range(len(in_pixels)):
                in_pix = in_pixels.pop(0)
                if higher_numbers_placed < higher_number_count:
                    for j in range(higher_number):
                        out_pix_outdex = random.randint(0,len(out_pixels)-1)
                        out_pix = out_pixels.pop(out_pix_outdex)
                        self.pairings.append([in_pix,out_pix])
                    higher_numbers_placed += 1
                else:
                    for j in range(higher_number-1):
                        out_pix_outdex = random.randint(0,len(out_pixels)-1)
                        out_pix = out_pixels.pop(out_pix_outdex)
                        self.pairings.append([in_pix,out_pix])
        self.best_score = self.get_score(self.pairings)
        self.last_best_score = self.best_score
    def get_score(self,pairings):
        aggregate = 0
        for pair in pairings:
            delta_x = pair[1][0]-pair[0][0]
            delta_y = pair[1][1]-pair[0][1]
            distance = (delta_x**2+delta_y**2)**.5
            aggregate += distance
        return aggregate
    def mutate(self):
        self.generation_count += 1
        population = [copy.deepcopy(self.pairings)]
        for _ in range(1,self.population_size):
            new_pairings = list(self.pairings)
            for i in range(len(new_pairings)):
                if random.random() < self.mutation_rate:
                    trade_partner = random.randint(0,len(new_pairings)-1)
                    if self.relationship in ("one to many","one to one"):
                        temp = new_pairings[trade_partner][1]
                        new_pairings[trade_partner][1]=tuple(new_pairings[i][1])
                        new_pairings[i][1]=tuple(temp)
                    else:
                        temp = new_pairings[trade_partner][0]
                        new_pairings[trade_partner][0]=tuple(new_pairings[i][0])
                        new_pairings[i][0]=tuple(temp)
            population.append(copy.deepcopy(new_pairings))
        #
        scores = [self.get_score(pairing) for pairing in population]
        best_score_index = scores.index(min(scores))
        self.pairings = copy.deepcopy(population[best_score_index])
        self.best_score = min(scores)
        if self.best_score == self.last_best_score:
            self.same_score_count += 1
            if self.same_score_count == self.convergence_threshold:
                self.converged = True
            print("    Generation " + str(self.generation_count) + "; "+ str(self.same_score_count) + " run;",end="\r")
        elif self.best_score < self.last_best_score:
            self.same_score_count = 0
            print("    Generation " + str(self.generation_count) + "; " + str(self.same_score_count) + " run;   best score: " + str(self.best_score),end="          \r")
        self.last_best_score = self.best_score
    def list_print(self,list):
        for item in list: 
            if "'list'" in str(type(item)) or "'tuple'" in str(type(item)):
                if "'list'" in str(type(item[0])) or "'tuple'" in str(type(item[0])):
                    if "'list'" in str(type(item[0][0])) or "'tuple'" in str(type(item[0][0])):
                        print("\n",end="")
                        self.list_print(item)
                    else:
                        print(item)
                else:
                    print(item)
            else:
                print(item)
    def generate_frames(self):
        print("Generating frames")
        # save first image
        number_str = "0"
        max_number_len = len(str(self.number_of_frames))
        while len(number_str) < max_number_len:
            number_str = "0"+number_str
        self.im1.save("frames/frame_"+number_str+".png")
        
        # save rest of images except for last
        print("Frames 1      /   " + str(self.number_of_frames),end="\r")
        for i in range(1,self.number_of_frames):
            print("    Frames "+ str(i+1),end="\r")
            ratio = i/self.number_of_frames
            #
            start_points = [item[0] for item in self.pairings]
            start_points = np.array(start_points)
            start_colors = self.np_im1[tuple(start_points.T.tolist())].astype(np.float32)
            end_points   = [item[1] for item in self.pairings]
            end_points   = np.array(end_points)
            end_colors   = self.np_im2[tuple(end_points.T.tolist())].astype(np.float32)
            #
            point_deltas = end_points - start_points
            point_deltas = point_deltas * ratio
            color_deltas = end_colors - start_colors
            color_deltas = color_deltas * ratio
            #
            current_points = start_points + point_deltas
            current_colors = start_colors + color_deltas
            #
            frame = Image.new('RGB',self.im1.size,(255, 255, 255))
            draw = ImageDraw.Draw(frame)
            for j in range(len(start_points)-1):
                location = current_points[j]
                # print("\nlocation: " + str(location))
                location[0],location[1]=location[1],location[0]
                location = tuple(location)
                # input("location: " + str(location))
                color = current_colors[j].astype(np.int32)
                color = tuple(color)
                # print("color: " + str(color))
                color_as_hex_string ='#%02x%02x%02x' % color
                # input("color_as_hex_string: " + str(color_as_hex_string))
                draw.point(location,color_as_hex_string)
            
            number_str = str(i)
            max_number_len = len(str(self.number_of_frames))
            while len(number_str) < max_number_len:
                number_str = "0"+number_str
            frame_name = "frames/frame_" + number_str +".png"
            frame.save(frame_name)
        
        # save last image
        self.im2.save("frames/frame_"+str(self.number_of_frames)+".png")
        print("\n",end="")
    def render_video(self):
        # command = "ffmpeg -framerate 60 -i frames/frame_%03d.png -vf tpad=stop_mode=clone:stop_duration=.5 output.mp4"
        # ffmpeg -framerate 60 -i frames/frame_%02d.png -vf tpad=stop_mode=clone:stop_duration=0.1 output.mp4
        command = "ffmpeg -framerate " + str(self.frames_per_second) +" -i frames/frame_%02d.png -vf tpad=stop_mode=clone:stop_duration=0.1 output.mp4 -y"
        subprocess.call(command,shell=True)
            
        

im1 = "test_images/1+1.png"
im2 = "test_images/2.png"
if __name__ == '__main__':
    transformer = Transformer(im1,im2)