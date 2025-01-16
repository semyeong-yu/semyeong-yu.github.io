import os
import cv2

def change_imgname(root_dir):
    dirs = os.listdir(root_dir)
    for dirr in dirs:
        dir_path = os.path.join(root_dir, dirr)
        if os.path.isdir(dir_path):
            files = os.listdir(dir_path)
            for file in files:
                if file.endswith('.png') or file.endswith('.PNG') or file.endswith('.jpg') or file.endswith('.JPG'):
                    new_file = file.rsplit('.', 1)[0] + 'm.PNG'
                    os.rename(os.path.join(dir_path, file), os.path.join(dir_path, new_file))

if __name__ == "__main__":
    change_imgname("./")