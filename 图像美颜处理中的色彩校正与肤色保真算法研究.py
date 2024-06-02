import cv2  
import numpy as np  
import matplotlib.pyplot as plt


#色彩平衡
  
def simple_color_balance(image, r_factor=1.0, g_factor=1.0, b_factor=1.0):  
    """  
    简单的色彩平衡函数，通过调整RGB通道的因子来改变图像的颜色。  
      
    :param image: 输入图像，应为numpy数组，数据类型为uint8，通道顺序为BGR。  
    :param r_factor: 红色通道的缩放因子。  
    :param g_factor: 绿色通道的缩放因子。  
    :param b_factor: 蓝色通道的缩放因子。  
    :return: 调整后的图像。  
    """  
    # 将图像数据转换为float32以避免整数溢出  
    float_image = np.float32(image)  
  
    # 调整RGB通道  
    float_image[:, :, 0] *= r_factor  # 红色通道  
    float_image[:, :, 1] *= g_factor  # 绿色通道  
    float_image[:, :, 2] *= b_factor  # 蓝色通道  
  
    # 确保像素值在0-255范围内  
    float_image = np.clip(float_image, 0, 255)  
  
    # 将数据转换回uint8  
    balanced_image = np.uint8(float_image)  
  
    return balanced_image  
  
# 读取图像  
image = cv2.imread('histogram_equalization/blue.jpg')  
  
# 应用色彩平衡（此处可根据实际情况调整数值）  
balanced_image = simple_color_balance(image, r_factor=0.65, g_factor=1.05, b_factor=1.4)  
  
# 显示原始图像和平衡后的图像  
cv2.imshow('Original Image', image)  
cv2.imshow('Color Balanced Image', balanced_image)  
cv2.waitKey(0)  
cv2.destroyAllWindows()

# 保存校正后的图像  
cv2.imwrite('color_balanced_image.jpg', balanced_image)


#直方图均衡


# 导入原始图像,色彩空间为灰度图  
src_img = cv2.imread('histogram_equalization/rose.png', 0)  
# 调用cv2.calcHist 函数绘制直方图  
img_hist = cv2.calcHist([src_img], [0], None, [256], [0, 256])[0]  # 取第一个通道（灰度图只有一个通道）  
  
# 直方图均衡化,调用cv2.equalizeHist 函数实现  
result_img = cv2.equalizeHist(src_img)  
  
# 显示原始图像  
cv2.imshow('src_img', src_img)  
# 显示均衡化后的图像  
cv2.imshow('result_img', result_img)  
cv2.waitKey(0)  
cv2.destroyAllWindows()  # 添加这行代码来关闭所有窗口  
  
# 用蓝色绘制原始图像直方图  
plt.figure(figsize=(10, 5))  
plt.bar(range(256), img_hist, color="b", alpha=0.7, label='Original Histogram')  
plt.xlim([0, 256])  
plt.title('Histogram for Original Image')  
plt.xlabel('Pixel Values')  
plt.ylabel('Frequency')  
plt.legend(loc='upper right')  
plt.show()  
  
# 绘制均衡化后的直方图  
result_hist = cv2.calcHist([result_img], [0], None, [256], [0, 256])[0]  
plt.figure(figsize=(10, 5))  
plt.bar(range(256), result_hist, color="g", alpha=0.7, label='Equalized Histogram')  
plt.xlim([0, 256])  
plt.title('Histogram for Equalized Image')  
plt.xlabel('Pixel Values')  
plt.ylabel('Frequency')  
plt.legend(loc='upper right')  
plt.show()





#光线补偿



def find_top_percent_brightness(img, percent):  
    # 将图像转换为灰度图  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    # 展平图像为一维数组  
    flat = gray.flatten()  
    # 对像素值进行排序  
    sorted_indices = np.argsort(flat)[::-1]  
    # 找到前percent%的像素数量的索引  
    num_pixels = gray.size  
    num_top_pixels = int(num_pixels * percent)  
    # 获取这些像素的灰度值  
    top_pixel_values = flat[sorted_indices[:num_top_pixels]]  
    # 计算平均灰度值  
    ave_gray = np.mean(top_pixel_values)  
    return ave_gray  
  
def light_compensation(img, percent=0.05):  
    # 找到前5%的亮度像素的平均灰度值  
    ave_gray = find_top_percent_brightness(img, percent)  
    # 如果平均灰度值已经是255，则不需要进行补偿  
    if ave_gray >= 255:  
        return img  
      
    # 计算光照补偿系数  
    coe = 255.0 / ave_gray  
      
    # 对图像的每个通道进行光照补偿  
    img_compensated = np.clip(img * coe, 0, 255).astype(np.uint8)  
      
    return img_compensated  



#肤色模型建立


# 读取图像  
img = cv2.imread('c:/Users/86157/Desktop/照片/证件照/证件照白底2024.jpg')  
# 对图像进行光照补偿  
img_compensated = light_compensation(img)  


  
# 读取图像  
image = cv2.imread('c:/Users/86157/Desktop/照片/证件照/证件照白底2024.jpg')  
  
# 将RGB图像转换为YCbCr颜色空间  
ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)  
  
# 提取亮度分量Y（在YCrCb中索引为0）  
Y, _, _ = cv2.split(ycbcr_image)  
  
# 假设使用亮度分量Y进行阈值分割  
# 这里需要根据你的图像来选择一个合适的阈值  
_, threshold_image = cv2.threshold(Y, 128, 255, cv2.THRESH_BINARY)  
  
# 应用中值滤波来减少噪声  
# 使用3x3的中值滤波器，可以根据需要调整大小  
filtered_image = cv2.medianBlur(threshold_image, 5)  
  
# 使用findContours函数查找轮廓  
# 注意：OpenCV 3和4的findContours返回不同的值，这里假设使用OpenCV 4  
contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
  
# 创建一个与原始图像相同大小的空图像来保存简化后的特征区域  
simplified_image = np.zeros_like(image)  
  
# 遍历每个轮廓，并基于高宽比和面积进行简化  
for contour in contours:  
    # 计算轮廓的边界框  
    x, y, w, h = cv2.boundingRect(contour)  
      
    # 计算高宽比  
    aspect_ratio = float(w) / h  
      
    # 计算面积  
    area = cv2.contourArea(contour)  
      
    # 设置高宽比和面积的阈值  
    aspect_ratio_threshold = 0.5  
    area_threshold = 100  
      
    # 如果轮廓满足条件，则在simplified_image中绘制它  
    if aspect_ratio > aspect_ratio_threshold and area > area_threshold:  
        cv2.rectangle(simplified_image, (x, y), (x + w, y + h), (255, 255, 255), -1)  
  



#肤色分割技术

  
# 读取图像  
image = cv2.imread('c:/Users/86157/Desktop/照片/证件照/证件照白底2024.jpg')  
  
# 转换为YCrCb颜色空间  
ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)  
  
# 提取Cr和Cb通道  
cr, cb, _ = cv2.split(ycrcb_image)  
  
# 设定肤色范围（这个范围需要根据实际情况调整）  
# 假设的肤色范围（仅作示例）  
lower_cr = np.array([133, 77, 77])  
upper_cr = np.array([173, 127, 127])  
  
# 创建一个肤色掩码  
skin_mask = cv2.inRange(ycrcb_image, lower_cr, upper_cr)  
  
# 对肤色掩码进行中值滤波，以减少噪声  
skin_mask = cv2.medianBlur(skin_mask, 5)  
  
# 形态学操作，去除小的非肤色区域  
kernel = np.ones((3, 3), np.uint8)  
skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)  
skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)  
  
# 找到面部轮廓  
contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
  
# 假设最大轮廓是面部  
c = max(contours, key=cv2.contourArea)  
  
# 创建一个与原图同样大小的空白图像  
face_mask = np.zeros_like(image)  
  
# 绘制面部轮廓  
cv2.drawContours(face_mask, [c], -1, (255, 255, 255), thickness=-1)  
  
# 面部区域转换为二值图像  
face_binary = cv2.bitwise_and(image, image, mask=face_mask)  
gray_face = cv2.cvtColor(face_binary, cv2.COLOR_BGR2GRAY)  
_, face_binary_thresh = cv2.threshold(gray_face, 128, 255, cv2.THRESH_BINARY_INV)  
  
# 找到二值图像中的轮廓  
eye_contours, _ = cv2.findContours(face_binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
  
# 假设眼睛是面部区域内两个最大的轮廓  
eyes = sorted(eye_contours, key=cv2.contourArea, reverse=True)[:2]  
  
# 绘制眼睛轮廓（可选）  
for eye in eyes:  
    M = cv2.moments(eye)  
    if M["m00"] != 0:  
        cX = int(M["m10"] / M["m00"])  
        cY = int(M["m01"] / M["m00"])  
        cv2.circle(image, (cX, cY), 3, (0, 255, 0), -1)  
  

#肤色分析

  
# 假设的阈值函数，用于根据Y值计算Cb和Cr的阈值  
# 这里的函数是示意性的，你需要根据实际数据来定义它  
def get_skin_color_thresholds(y):  
    # 这里是一个简单的线性模型，你可能需要更复杂的模型或查找表  
    cb_lower = 77 + (1 - y / 255) * 10  # 假设的Cb下界  
    cb_upper = 127 - (1 - y / 255) * 10  # 假设的Cb上界  
    cr_lower = 133 + (1 - y / 255) * 15  # 假设的Cr下界  
    cr_upper = 173 - (1 - y / 255) * 15  # 假设的Cr上界  
    return (cb_lower, cb_upper), (cr_lower, cr_upper)  
  
# 读取图像  
image = cv2.imread('c:/Users/86157/Desktop/照片/证件照/证件照白底2024.jpg')  
  
# 转换到YCbCr颜色空间  
ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)  
  
# 分割Y, Cb, Cr通道  
y, cb, cr = cv2.split(ycbcr)  
  
# 初始化肤色掩码  
skin_mask = np.zeros(image.shape[:2], dtype=np.uint8)  
  
# 遍历图像的每个像素  
for i in range(y.shape[0]):  
    for j in range(y.shape[1]):  
        # 获取当前像素的Y值  
        y_val = y[i, j]  
          
        # 根据Y值获取Cb和Cr的阈值  
        cb_thresh, cr_thresh = get_skin_color_thresholds(y_val)  
          
        # 检查当前像素的Cb和Cr值是否在阈值范围内  
        if cb_thresh[0] <= cb[i, j] <= cb_thresh[1] and cr_thresh[0] <= cr[i, j] <= cr_thresh[1]:  
            # 肤色像素，将对应位置的肤色掩码设为255  
            skin_mask[i, j] = 255  
  
# 将肤色掩码应用到原图像上，用于可视化  
skin_image = cv2.bitwise_and(image, image, mask=skin_mask)  
  



#自适应阈值分割

  
# 假设函数，用于根据Y值返回Cb和Cr的均值及标准差  
# 在实际应用中，这些函数需要根据你的数据集进行拟合  
def get_cb_mean_std(y):  
    # 假设的Cb均值和标准差函数（这里仅为示例，需要替换为真实拟合的函数）  
    cb_mean = 100 + y / 255 * 10  # 示例：随着Y增加，Cb均值也增加  
    cb_std = 15 - y / 255 * 5     # 示例：随着Y增加，Cb标准差减少  
    return cb_mean, cb_std  
  
def get_cr_mean_std(y):  
    # 假设的Cr均值和标准差函数（同样需要替换为真实拟合的函数）  
    cr_mean = 150 - y / 255 * 10  # 示例：随着Y增加，Cr均值减少  
    cr_std = 20 + y / 255 * 3     # 示例：随着Y增加，Cr标准差增加  
    return cr_mean, cr_std  
  
# 肤色检测的范围参数，这里取2.5  
alpha = 2.5  
  
# 读取图像  
image = cv2.imread('c:/Users/86157/Desktop/照片/证件照/证件照白底2024.jpg')  
  
# 转换到YCbCr颜色空间  
ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)  
  
# 分割Y, Cb, Cr通道  
y, cb, cr = cv2.split(ycbcr)  
  
# 初始化肤色掩码  
skin_mask = np.zeros(image.shape[:2], dtype=np.uint8)  
  
# 遍历图像的每个像素  
for i in range(y.shape[0]):  
    for j in range(y.shape[1]):  
        # 获取当前像素的Y值  
        y_val = y[i, j]  
          
        # 根据Y值计算Cb和Cr的均值及标准差  
        cb_mean, cb_std = get_cb_mean_std(y_val)  
        cr_mean, cr_std = get_cr_mean_std(y_val)  
          
        # 计算肤色检测的范围  
        cb_lower = cb_mean - alpha * cb_std  
        cb_upper = cb_mean + alpha * cb_std  
        cr_lower = cr_mean - alpha * cr_std  
        cr_upper = cr_mean + alpha * cr_std  
          
        # 检查当前像素的Cb和Cr值是否在肤色检测范围内  
        if cb_lower <= cb[i, j] <= cb_upper and cr_lower <= cr[i, j] <= cr_upper:  
            # 肤色像素，将对应位置的肤色掩码设为255  
            skin_mask[i, j] = 255  
  
# 将肤色掩码应用到原图像上，用于可视化  
skin_image = cv2.bitwise_and(image, image, mask=skin_mask)  
  
# 显示结果  
cv2.imshow('Original Image', image)  
cv2.imshow('Skin Detection', skin_image)  
cv2.waitKey(0)  
cv2.destroyAllWindows()