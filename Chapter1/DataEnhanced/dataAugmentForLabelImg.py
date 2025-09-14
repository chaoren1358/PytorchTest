import os
import cv2
import matplotlib.pyplot as plt

class DataArgument:#@save
    def __init__(self):
        pass
        
    def readImg(self, imgPath, flag = 0):
        if os.path.exists(imgPath):
            if flag == 0:
                img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return None
                h, w, c = img.shape
                print(f"✅ oriimg读取图片成功: {imgPath}, 图片尺寸: {h}x{w}x{c}")
                cv2.imwrite('oriimg.jpg', img)
                return img
            if flag == 1 or flag == 2:
                img = cv2.imread(imgPath)
                if img is None:
                    return None
                colorImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if colorImg is None:
                    return None
                if flag == 1:
                    cv2.imwrite('colorImg.jpg', colorImg)
                    h, w, c = colorImg.shape
                    print(f"✅ colorImg读取图片成功: {imgPath}, 图片尺寸: {h}x{w}x{c}")
                    return colorImg
                else:
                    grayImg = cv2.cvtColor(colorImg, cv2.COLOR_RGB2GRAY)
                    if grayImg is None:
                        return None
                    cv2.imwrite('gray.jpg', grayImg)
                    h, w = grayImg.shape
                    print(f"✅ grayImg读取图片成功: {imgPath}, 图片尺寸: {h}x{w}")
                    return grayImg
        else:
            print(f"❌ 文件不存在: {imgPath}")
        return None

    def showPic(self, img, bboxes=None):
        if bboxes is not None:
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = bbox[2]
                y_max = bbox[3]
                cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255 , 0), 3)
        plt.imshow(img)
        plt.axis('off')  # 关闭坐标轴
        plt.show()
        # cv2.namedWindow('pic', 0)
        # cv2.moveWindow('pic', 0, 0)
        # cv2.resizeWindow('pic', 1200, 800)
        # cv2.imshow('pic', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()