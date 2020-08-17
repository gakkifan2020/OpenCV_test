import numpy as np
import cv2
import time
from split_ADT_pic_to_three import t
# import matplotlib.pyplot as plt
def split_up(pic):
    BoundingAreaThres = 80000.0
    # 上下边界截取方框阈值
    thres_param_1 = 30
    # 上边界高阈值
    upper_thres = 40
    upper_second_thres = 190
    down_thres = 120

    path = './dataset/orign/'
    # image = cv2.imread("reko.jpg", cv2.IMREAD_COLOR)
    image = cv2.imread(path + pic, cv2.IMREAD_GRAYSCALE)
    img_color1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img_color2 = np.copy(img_color1)
    # cv2.imshow("show", image)
    # cv2.waitKey(0)
    # 步骤一 ： 开运算
    #            滤波（方框滤波，高斯滤波，blur， bilateral滤波， 中值滤波）
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    # cv2.imwrite("dataset/upper_part/{}.png".format(num), blur)

    #            腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.erode(blur, kernel)       # 腐蚀
    dst1 = cv2.dilate(dst, kernel)     # 膨胀

    # 步骤二 ： 上边沿阈值分割
    # 上边界阈值
    ret, thresh = cv2.threshold(dst1, thres_param_1, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("upper_record\\2_ero_dil.png", thresh)
    # cv2.imshow("show1", thresh)
    # cv2.waitKey(0)

    # 步骤三 ： 高分割阈值边界矩形
    #          提取最大轮廓
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    hierarchy = np.squeeze(hierarchy)
    box = []
    max_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])

        if area < BoundingAreaThres:
            continue
        # 矩形四个角点取整
        # rect 格式如右所示 —— ((646.2275390625, 236.74612426757812), (868.9734497070312, 247.57138061523438), -0.2270626723766327)
        # 中心坐标 x ， y ， 宽度 ， 高度 ， 角度
        rect = cv2.minAreaRect(contours[i])
        # 转化成四个坐标， 存入box
        box = np.int0(cv2.boxPoints(rect))

        cv2.drawContours(img_color2, [box], 0, (255, 0, 0), 2)

    # cv2.imwrite("upper_record\\box.png", img_color2)
    # cv2.imshow("show2", img_color2)
    # cv2.waitKey(0)

    #  图像上边界分割
    # 由高阈值分割图寻找上边界
    image_width = img_color1.shape[1]
    image_height = img_color1.shape[0]
    # 像素清零
    left = min(box[:, 0])
    right = max(box[:, 0])
    top = min(box[:, 1])
    down = max(box[:, 1])

    left_corner_y = [200, 0]
    left_corner_x = [0, 550]
    left_corner_line = np.polyfit(left_corner_x, left_corner_y, 2, full=True)
    left_corner_weights = left_corner_line[0]

    right_corner_y = [340, 0]
    right_corner_x = [1400, 650]
    right_corner_line = np.polyfit(right_corner_x, right_corner_y, 2, full=True)
    right_corner_weights = right_corner_line[0]

    for row in range(image_height):
        for col in range(image_width):
            if row < top or row > down or col < left or col > right:
                img_color1[row][col] = (0, 0, 0)
            if row < np.polyval(left_corner_weights, col):
                img_color1[row][col] = (0, 0, 0)
            if row < np.polyval(right_corner_weights, col):
                img_color1[row][col] = (0, 0, 0)

    # cv2.imwrite("upper_record\\box.png", img_color1)
    # cv2.imshow("show3", img_color1)
    # cv2.waitKey(0)

    upper_boundary = []
    upper_second_boundary = []

    # 从上往下提取边界
    for col in range(left, right, 1):
        for row in range(top, down, 1):
            if img_color1[row][col][0] > upper_thres:
                upper_boundary.append([row, col])
                break

    # 拟合上曲线

    row_up = []
    col_up = []
    # image = np.copy(img_color1)
    for i in range(len(upper_boundary)):
        row_up.append(upper_boundary[i][0])
        col_up.append(upper_boundary[i][1])
    row_up = np.array(row_up)
    col_up = np.array(col_up)

    result= np.polyfit(col_up, row_up, 3, full=True)
    weights_up = result[0]
    residual = result[1]/len(upper_boundary)

    # new_row_up = np.polyval(weights,col_up)

    # 上边界以上像素清零

    for i in range(top, down, 1):
        for j in range(left, right, 1):
            if i < np.polyval(weights_up, j):
                img_color1[i][j] = 0

    # cv2.imshow("show4", img_color1)
    # cv2.waitKey(0)


    # 由高阈值分割图寻找上部第二层边界

    # 从上往下提取第二边界
    for col in range(left, right, 1):
        for row in range(top, down, 1):
            if img_color1[row][col][0] > upper_second_thres:
                upper_second_boundary.append([row, col])
                break

    # 拟合上部第二条曲线

    row_down = []
    col_down = []
    # image = np.copy(img_color1)
    for i in range(len(upper_second_boundary)):
        row_down.append(upper_second_boundary[i][0])
        col_down.append(upper_second_boundary[i][1])
    row_down = np.array(row_down)
    col_down = np.array(col_down)

    result= np.polyfit(col_down, row_down, 3, full=True)
    weights_down = result[0]
    residual = result[1]/len(upper_second_boundary)

    # new_row_down = np.polyval(weights,col_down)

    # 上部第二条边界以下像素清零

    for i in range(top, down, 1):
        for j in range(left, right, 1):
            if i > np.polyval(weights_down, j):
                img_color1[i][j] = 0

    # cv2.imwrite("upper_record\\upper_1.png", img_color1)
    # cv2.imshow("show5", img_color1)
    # cv2.waitKey(0)
    cv2.imwrite("dataset/upper_part/{}.png".format(pic), img_color1)
    print('time cost:', time.time() - t, pic)
    return True



if __name__ == '__main__':
    split_up('./dataset/orign/1.png')


    #
    # # 从左往右提取边界
    # for row in range(image_height):
    #     for col in range(image_width):
    #         if img_color1[row][col][0] > upper_thres:
    #             left_boundary.append([row,col])
    #             break
    #
    # # 从右往左提取边界
    # for row in range(image_height-1, -1, -1):
    #     for col in range(image_width-1, -1, -1):
    #         if img_color1[row][col][0] > upper_thres:
    #             right_boundary.append([row,col])
    #             break

    # plot1 = plt.plot(col,row,'*',label='original values')
    # plot2 = plt.plot(col,new_row,'r',label='polyfit values')
    #
    # plt.xlabel('xaxis')
    # plt.ylabel('yaxis')
    #
    # plt.legend(loc=4)  #指定legend的位置,读者可以自己help它的用法
    #
    # plt.title('polyfitting')
    # plt.savefig('p1.png')
    # plt.show()
    # cv2.waitKey(0)








    # image = np.copy(img_color1)
    #
    # N = len(upper_boundary)
    # X = np.zeros((len(upper_boundary[0])+1, len(upper_boundary[0])+1),  dtype=np.uint8)
    # Y = np.zeros((len(upper_boundary[0])+1, 1), dtype=np.uint8)
    # # 构造矩阵 X
    # for row in range(len(upper_boundary[0])+1):
    #     for col in range(len(upper_boundary[0])+1):
    #         for i in range(N):
    #             X[row][col] = X[row][col] + np.power(upper_boundary[i][0], row + col)
    # # 构造矩阵 Y
    # for row in range(len(upper_boundary[0]) + 1):
    #     for i in range(N):
    #         Y[row] = Y[row] + np.power(upper_boundary[i][0], row) * upper_boundary[i][1]
    #
    # # 构造向量 A
    # A = np.zeros((len(upper_boundary[0])+1, 1), dtype=cv2.CV_64F)
    # # A = np.zeros((len(upper_boundary[0])+1, 1), dtype=np.uint8)
    #
    #
    # # 求解 A
    # cv2.solve(X, Y, A, flags=cv2.DECOMP_LU)
    #
    # cv2.imshow("show", image)
    # cv.imwrite("D:/curve.png", image)

    # cv2.imshow("show", image)
    # cv.imwrite("D:/curve.png", image)

    # poly = np.poly1d(np.polyfit(x, y, 3))
    # print(poly)
    # for t in range(30, 250, 1):
    #     y_ = np.int(poly(t))
    #     cv.circle(image, (t, y_), 1, (0, 0, 255), 1, 8, 0)
    # cv.imshow("fit curve", image)
    # cv.imwrite("D:/fitcurve.png", image)




    # ***  Soble  算子
    # x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    # y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
    # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
    # Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
    # Scale_absY = cv2.convertScaleAbs(y)
    # soble = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)

