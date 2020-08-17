import OpenCV_spllit_down
# import OpenCV_spllit_middle
import OpenCV_spllit_up
import os
from multiprocessing import process
from multiprocessing import Pool

which = 0 # 0:up  1: middle 2: down

path = './dataset/orign/'
image_files = os.listdir(path)  # 扫描目标路径的文件,将文件名存入列表



a = 0

if which == 0:
    if __name__ == '__main__':
        p = Pool(4)
        length = len(image_files)//4
        for i in range(4):
            image_files_i = image_files[i*length:(i+1)*length]
            p.apply_async(OpenCV_spllit_up.process_pic_list, args=(path, image_files_i))
            p.close()
            p.join()
    # OpenCV_spllit_up.process_pic_list(path, image_files_1)


if which == 2:
    for pic in image_files:
        OpenCV_spllit_down.split_up( path + pic, a)
        a += 1



print("finished")


