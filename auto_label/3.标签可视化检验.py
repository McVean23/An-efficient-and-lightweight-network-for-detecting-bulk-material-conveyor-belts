# import cv2
# import os
#
# img_folder = 'D:/Study/pycharm_projects/ultralytics/test_img'
# label_folder = 'D:/Study/pycharm_projects/ultralytics/test_label'
#
# # Get a list of label files in the folder
# label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
#
# if len(label_files) == 0:
#     print('No label files found in the folder:', label_folder)
# else:
#     # Select a label file
#     print('Available label files:')
#     for i, label_file in enumerate(label_files):
#         print(f'{i + 1}. {label_file}')
#
#     choice = input('Enter the number of the label file to visualize (1-%d): ' % len(label_files))
#     choice = int(choice) - 1
#
#     if choice < 0 or choice >= len(label_files):
#         print('Invalid choice!')
#     else:
#         label_file = label_files[choice]
#
#         # Construct label and image file paths
#         label_path = os.path.join(label_folder, label_file)
#         file_name = os.path.splitext(label_file)[0]
#         img_path = os.path.join(img_folder, file_name + '.jpg')
#
#         # Read image
#         img = cv2.imread(img_path)
#
#         # Set window name
#         window_name = file_name
#
#         # Resize the window
#         cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(window_name, 1280, 960)
#
#         # Read label file
#         with open(label_path, 'r') as f:
#             lines = f.readlines()
#
#         for line in lines:
#             line = line.strip()
#             if line:
#                 points = line.split(' ')
#                 points = [float(p) for p in points[1:]]  # Exclude the first '0' label
#                 num_points = len(points) // 2
#
#                 for i in range(num_points):
#                     x = int(points[i * 2] * img.shape[1])
#                     y = int(points[i * 2 + 1] * img.shape[0])
#
#                     cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
#
#         # Display the image
#         cv2.imshow(window_name, img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

import cv2
import os

def visualize_and_save(img_folder, label_folder, save_folder):
    # 获取标签文件夹中的所有标签文件
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    for label_file in label_files:
        # 构建标签和图像文件路径
        label_path = os.path.join(label_folder, label_file)
        file_name = os.path.splitext(label_file)[0]
        img_path = os.path.join(img_folder, file_name + '.jpg')

        # 读取图像
        img = cv2.imread(img_path)

        # 读取标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line:
                points = line.split(' ')
                points = [float(p) for p in points[1:]]  # 排除第一个 '0' 标签
                visualize_label(img, points)

        # 保存绘制坐标点后的图像
        save_path = os.path.join(save_folder, file_name + '_with_labels.jpg')
        cv2.imwrite(save_path, img)

def visualize_label(img, points):
    # 在图像上绘制标签点
    for i in range(len(points) // 2):
        x = int(points[i * 2] * img.shape[1])
        y = int(points[i * 2 + 1] * img.shape[0])
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

# 图像文件夹路径
img_folder = 'D:/Study/pycharm_projects/ultralytics/test_img'
# 标签文件夹路径
label_folder = 'D:/Study/pycharm_projects/ultralytics/test_label'
# 保存文件夹路径
save_folder = 'D:/Study/pycharm_projects/ultralytics/val_label'

# 确保保存文件夹存在
os.makedirs(save_folder, exist_ok=True)

# 可视化并保存图片
visualize_and_save(img_folder, label_folder, save_folder)