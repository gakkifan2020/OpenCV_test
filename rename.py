# import OpenCV_spllit_down
# import OpenCV_spllit_middle
# import OpenCV_spllit_up
import os

path = './dataset/orign/'


filename_list = os.listdir(path)  # 扫描目标路径的文件,将文件名存入列表

a = 0
for i in filename_list:
 used_name = path + filename_list[a]
 new_name = path + "{}.png".format(a)
 os.rename(used_name, new_name)
 print("文件%s重命名成功,新的文件名为%s" %(used_name, new_name))
 a += 1


