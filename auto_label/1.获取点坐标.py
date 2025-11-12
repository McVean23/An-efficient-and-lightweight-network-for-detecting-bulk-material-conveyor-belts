import cv2

# 定义一个变量来保存鼠标点击的坐标
clicked_points = []

# 回调函数，用于处理鼠标点击事件
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 在左键按下时记录鼠标坐标
        clicked_points.append((x, y))
        print(f"Clicked at coordinates: ({x}, {y})")

# 加载图片
image_path = "D:/Study/pycharm_projects/ultralytics/test_img/20231128142907445_546.jpg"  # 将路径替换为你的图片路径
image = cv2.imread(image_path)

# 检查图片是否成功加载
if image is None:
    print("Error: Unable to load the image.")
else:
    # 创建窗口并绑定鼠标事件回调函数
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        # 显示图片
        cv2.imshow("Image", image)

        # 等待用户按下键盘上的任意键
        key = cv2.waitKey(1) & 0xFF

        # 如果用户按下 'q' 键，退出循环
        if key == ord("q"):
            break

    # 销毁窗口
    cv2.destroyAllWindows()

# 打印最终的点击坐标
print("Final Clicked Points:", clicked_points)