from ultralytics import SAM
import cv2
import os
import numpy as np

def is_point_inside_polygon(x, y, polygon):
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0][0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n][0]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y

    return inside

# # Load a model
model = SAM('D:/Study/pycharm_projects/ultralytics/sam_l.pt')

img_folder = 'D:/Study/pycharm_projects/ultralytics/test_img'
output_folder = 'D:/Study/pycharm_projects/ultralytics/test_label'

# # 读取图片
# img_path = 'test_img/20231201105447_105019_16.jpg'
# img = cv2.imread(img_path)

# # 定义多边形坐标按照顺时针方向
# polygon = np.array([[548, 344], [1319, 354], [1513, 1028], [326, 1027]], np.int32)
# polygon = polygon.reshape((-1, 1, 2))

# # 画多边形
# cv2.polylines(img, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
#
# # 显示图片
# cv2.imshow('Image with Clockwise Quadrilateral', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of image files in the folder
image_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]

for image_file in image_files:
    # Construct image and txt file paths
    img_path = os.path.join(img_folder, image_file)
    file_name = os.path.splitext(image_file)[0]
    txt_file_path = os.path.join(output_folder, file_name + '.txt')

    # Read image and get its dimensions
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape

    corner_point = np.array([[575, 333], [1312, 342], [1480, 1114], [375, 1146]], np.int32)
    interior_point = np.array([[734, 464], [1200, 486], [939, 670], [646, 925], [1218, 921]], np.int32)

    # ROI
    # 找出y方向上的最小和最大值
    min_y = np.min(corner_point[:, 1])
    max_y = np.max(corner_point[:, 1])
    # 计算第二个和第四个等分点的y坐标值
    y_values = np.linspace(min_y, max_y, num=5)[1::2]  # 4等分，取第2和第4个值
    # 分别提取较小和较大的值
    min_y_value = np.min(y_values)
    max_y_value = np.max(y_values)

    # Run inference with bboxes prompt
    results = model(img, points=[corner_point[0], corner_point[1], corner_point[2], corner_point[3],
                                 interior_point[0], interior_point[1], interior_point[2], interior_point[3], interior_point[4]], device=0)  #[900, 370]    [700, 16, 1609, 986]
    result = results[0]

    masks = result.masks
    all_points = masks.xy

    # 定义多边形坐标按照顺时针方向
    polygon = corner_point
    polygon = polygon.reshape((4, 1, 2))

    # cv2.polylines(img, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)

    # 存储在多边形外的坐标点
    points_outside_polygon = []

    for array_points in all_points:
        for point in array_points:
            x, y = point
            if not is_point_inside_polygon(x, y, polygon) and min_y_value < y < max_y_value:  # ROI
                points_outside_polygon.append(point)

    # # 画外部的坐标点（绿色）
    # for point in points_outside_polygon:
    #     cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
    #
    # # 调整窗口大小
    # img = cv2.resize(img, (640, 640))
    #
    # # 显示图片
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # YOLO format label
    with open(txt_file_path, 'w') as f:
        f.write('0 ')
        for points_array in points_outside_polygon:
            x = points_array[0] / img_width
            y = points_array[1] / img_height
            f.write('{:.6f} {:.6f} '.format(x, y))
        # f.write('\n')

    # # YOLO format label
    # with open(txt_file_path, 'w') as f:
    #     for points in all_points:
    #         f.write('0 ')
    #         for point in points:
    #             x = point[0] / img_width
    #             y = point[1] / img_height
    #             f.write('{:.6f} {:.6f} '.format(x, y))
    #         f.write('\n')

    print('Points coordinates saved to:', txt_file_path)