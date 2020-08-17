import OpenCV_spllit_down
# import OpenCV_spllit_middle
import OpenCV_spllit_up
import os
from multiprocessing import process
from multiprocessing import Pool
import time

which = 0 # 0:up  1: middle 2: down

path = './dataset/orign/'
image_files = os.listdir(path)  # 扫描目标路径的文件,将文件名存入列表
t = time.time()


a = 0

if which == 0:
    if __name__ == '__main__':

        p = Pool(4)
        p.map(OpenCV_spllit_up.process_pic_list, image_files)
        p.close()
        p.join()
    # OpenCV_spllit_up.process_pic_list(path, image_files_1)


if which == 2:
    for pic in image_files:
        OpenCV_spllit_down.split_up( path + pic, a)
        a += 1



    print("finished")


