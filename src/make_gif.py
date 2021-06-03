import os
import imageio

directory = "./image"
file_type = "png"
save_gif_name = "result"
speed_sec = {'duration': 0.5}

images = []

for file_name in os.listdir(directory):
    if file_name.endswith(".{}".format(file_type)):
        file_path = os.path.join(directory,file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave("{}/{}.gif".format(directory,save_gif_name), images, **speed_sec)