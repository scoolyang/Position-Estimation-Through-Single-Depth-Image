import statistics
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import data_syn
import matplotlib.pyplot as plt; plt.ion()

# def main():
#     try:
#         pipeline = rs.pipeline()
#         pipeline.start()
#
#         while True:
#             frames = pipeline.wait_for_frames()
#             depth
# This function is used to find out corresbounding imgae data given camera index and image number from all data txt file
# Three Inputs: 1. input YOLO output txtfile
#               2. image from which camera holder (ex: infinite, nissan)
#               3. input image number (the six digits of each image)
#               4. number of features need to be save in the matrix (default is 8)

# Two Outputs: 1. input matrix coming from YOLO output file
#               2. object to idx dictionary

def find_sample_data(open_file_name, img_num, features):
    save_object_to_idx_path = 'image_similarity_object_to_idx.npy'
    if not os.path.exists(save_object_to_idx_path):
        object_to_idx = {}
        np.save(save_object_to_idx_path, object_to_idx)
    input_x_matrix = np.zeros(features).reshape(1, features)
    if open_file_name == argument['infiniti_yolooutput_dir']:
        # print(1)
        img_num_range_lb = 8
        img_num_range_ub = 14
        # Sample Test
        # img_num_range_lb = 21
        # img_num_range_ub = 26
    elif open_file_name == argument['nissan_yolooutput_dir']:
        img_num_range_lb = 23
        img_num_range_ub = 29
        # Sample Test
        # img_num_range_lb = 21
        # img_num_range_ub = 26
    with open(open_file_name) as file:
        data = file.readlines()
        new_image_flag = 0
        object_to_idx = np.load(save_object_to_idx_path)
        new_class_index = len(object_to_idx.item())
        for line in data:
            numbers = line.split()
            if numbers != []:
                if numbers[0][0:3] == 'exp' and numbers[0][img_num_range_lb:img_num_range_ub] == str(img_num):
                    # image_name = numbers[0][14:len(numbers[0])]
                    new_image_flag = 1
                    new_image_array = np.zeros([features]).reshape(1, features)
                elif numbers[0][0:3] == 'exp' and numbers[0][img_num_range_lb:img_num_range_ub] != str(img_num):
                    new_image_flag = 0
                if new_image_flag == 1 and numbers[0][0] != 'e':
                    test_digit = numbers[0]

                    if test_digit.isdigit():
                        detect_num = float(numbers[0])
                        x_min = float(numbers[1])
                        y_min = float(numbers[2])
                        width = float(numbers[3])
                        length = float(numbers[4])
                        class_label_string = numbers[5]
                        if class_label_string[-1] == ':':
                            class_label_string = class_label_string[0:(len(class_label_string) - 1)]
                        if class_label_string not in object_to_idx.item():
                            object_to_idx.item()[class_label_string] = new_class_index
                            np.save(save_object_to_idx_path, object_to_idx.item())
                            new_class_index += 1
                        cfs = numbers[6]
                        if cfs[-1] != '%':
                            cfs = numbers[7]
                    else:
                        detect_num += 1
                        class_label_string = numbers[0]
                        if class_label_string[-1] == ':':
                            class_label_string = class_label_string[0:(len(class_label_string) - 1)]
                        if class_label_string not in object_to_idx.item():
                            object_to_idx.item()[class_label_string] = new_class_index
                            np.save(save_object_to_idx_path, object_to_idx.item())
                            new_class_index += 1
                        cfs = numbers[1]
                        if cfs[-1] != '%':
                            cfs = numbers[2]

                    new_image_array[0, 0] = float(img_num)
                    new_image_array[0, 1] = detect_num
                    new_image_array[0, 2] = x_min
                    new_image_array[0, 3] = y_min
                    new_image_array[0, 4] = width
                    new_image_array[0, 5] = length
                    new_image_array[0, 6] = object_to_idx.item()[class_label_string]
                    new_image_array[0, 7] = float(cfs[0:2])
                    if cfs[0:3] == '100':
                        new_image_array[0, 7] = float(cfs[0:3])
                    input_x_matrix = np.concatenate((input_x_matrix, new_image_array), axis=0)

        input_x_matrix = np.delete(input_x_matrix, (0), axis=0)
    return input_x_matrix, object_to_idx

# This function used to find out desired object bouning box coordinates in an image
# Three Inputs: 1. input matrix generated from YOLO output txtfile
#               2. corresponding input image in rgb format
#               3. the desired number of object in the input image

# Four Outputs: 1. bounding box x axis min coordinate
#               2. bounding box x axis max coordinate
#               3. bounding box y axis min coordinate
#               4. bounding box y axis max coordinate

def find_coor(input_x_matrix, input_img, object_num):
    xmin = int(input_x_matrix[object_num][2])
    if xmin <= 0:
        xmin = 0
    xmax = xmin + int(input_x_matrix[object_num][4])
    ymin = int(input_x_matrix[object_num][3])
    ymax = ymin + int(input_x_matrix[object_num][5])
    if ymax >= input_img.shape[0]:
        ymax = input_img.shape[0]   #480
        # img_test.shape[1]  # 640
    return xmin, xmax, ymin, ymax

def depth_info_from_img(input_depth_img, input_x_matrix):
    object_num_in_img = input_x_matrix.shape[0]
    object_depth_median_pair = []
    object_center_pixel_coor_list = []

    for k in range(object_num_in_img):

        xmin, xmax, ymin, ymax = find_coor(input_x_matrix, input_depth_img, k)
        roi_img = input_depth_img[ymin: ymax, xmin: xmax]
        pixel_depth_value_list = []
        depth_pixel_coor_dict = {}

        for v in range(roi_img.shape[0]):
            for u in range(roi_img.shape[1]):
                dist = roi_img[v, u] / 1000
                if dist < argument['depth_max_range_value'] and dist > 0:
                    depth_pixel_coor_dict[(u, v)] = dist
                    pixel_depth_value_list.append(dist)
        if pixel_depth_value_list == []:
            pixel_depth_value_list.append(0)
        median = statistics.median(pixel_depth_value_list)
        # mode = statistics.mode(pixel_depth_value_list)
        median_pair = [k, median]
        # mode_pair = [k, mode]
        object_depth_median_pair.append(median_pair)
        # object_depth_mode_pair.append(mode_pair)

        # Special Method Take the center of the ROI object coordinate
        roi_img_center_v = np.ceil(roi_img.shape[0] / 2).astype(np.int16) + ymin
        roi_img_center_u = np.ceil(roi_img.shape[1] / 2).astype(np.int16) + xmin
        roi_img_center_d = input_depth_img[roi_img_center_v, roi_img_center_u] / 1000
        pixel_coor = [roi_img_center_u, roi_img_center_v, roi_img_center_d]
        object_center_pixel_coor_list.append(pixel_coor)

        # cv2.imshow("Image", roi_img)
        # cv2.waitKey(1000)
        # cv.destroyAllWindows()
    return object_depth_median_pair, object_center_pixel_coor_list

def image_test(input_test_image):

    img_center_v = np.ceil(input_test_image.shape[0] / 2).astype(np.int16)
    img_center_u = np.ceil(input_test_image.shape[1] / 2).astype(np.int16)
    img_center_d = input_test_image[img_center_v, img_center_u] / 1000
    pixel_coor = [img_center_u, img_center_v, img_center_d]

    return pixel_coor

def pixel_to_optical(pixel_coor, intrinsic_mat_inv):
    pixel_coor_depth_frame = np.array([pixel_coor[0], pixel_coor[1], 1])
    temp_mat = (intrinsic_mat_inv@pixel_coor_depth_frame) * pixel_coor[2]
    # optical_z = np.sqrt(temp_mat[2])
    # print(optical_z)
    optical_z = temp_mat[2]
    optical_x = temp_mat[0]
    optical_y = temp_mat[1]
    optical_coor = np.array([optical_x, optical_y, optical_z])
    return optical_coor

def optical_to_world(optical_coor, current_gps_pose, depth_camera_pos_ori):
    Final_mat = np.eye(4)
    pos_wc_mat_x = current_gps_pose[0] + depth_camera_pos_ori[0]
    pos_wc_mat_y = current_gps_pose[1] + depth_camera_pos_ori[1]
    pos_wc_mat_z = depth_camera_pos_ori[2]                   #unit meters
    pos_wc_mat = np.array([pos_wc_mat_x, pos_wc_mat_y, pos_wc_mat_z])

    R_wc_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R_oc_mat = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    temp1 = np.matmul(R_oc_mat, np.transpose(R_wc_mat))
    temp2 = np.matmul(-1*R_oc_mat, np.transpose(R_wc_mat))
    temp3 = np.matmul(temp2, pos_wc_mat).reshape(3, 1)

    Final_mat[0:3, 0:3] = temp1
    Final_mat[0, 3] = temp3[0]
    Final_mat[1, 3] = temp3[1]
    Final_mat[2, 3] = temp3[2]

    optical_coor = np.concatenate((optical_coor.reshape(3, 1), [[1]]), axis=0)
    # print(optical_coor)
    world_coor = np.matmul(np.linalg.inv(Final_mat), optical_coor)
    # pos_wc_mat_s = np.concatenate((pos_wc_mat, [[1]]), axis=0)
    # world_coor = np.ceil((np.subtract(world_coor, pos_wc_mat_s)) / MAP['res']).astype(np.int16)
    # world_coor[0, :] = current_best_pose[0] + world_coor[0, :]
    # world_coor[1, :] = current_best_pose[1] + world_coor[1, :]
    return  world_coor

def local_car_frame_to_earth_frame(local_position, longtitude_roll_phi, latitude_yaw_lamda):
    local_pos = np.array([local_position[0][0], local_position[1][0], 1]).reshape(3, 1)
    print(local_pos)
    longtitude_roll_phi_rad = longtitude_roll_phi * np.pi / 180
    latitude_yaw_lamda_rad = -latitude_yaw_lamda * np.pi / 180
    roll_rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(longtitude_roll_phi_rad), -np.sin(longtitude_roll_phi_rad)],
                                    [0, np.sin(longtitude_roll_phi_rad), np.cos(longtitude_roll_phi_rad)]])
    yaw_rotation_matrix = np.array([[np.cos(latitude_yaw_lamda_rad), -np.sin(latitude_yaw_lamda_rad), 0],
                                    [np.sin(latitude_yaw_lamda_rad), np.cos(latitude_yaw_lamda_rad), 0],
                                    [0, 0, 1]])

    print(roll_rotation_matrix)
    earth_pos = roll_rotation_matrix @ yaw_rotation_matrix @ local_pos
    earth_pos_list = [earth_pos[0][0], earth_pos[1][0]]
    return earth_pos_list

def main():
    depth_camera_pos_ori = np.array([[0.9144], [0.254], [0.6858]])    # unit m
    cam1 = data_syn.read_cameratimestamp('B:/SRIP_2019/depth_position_test/CameraTimestamp_infiniti_test3.csv')
    cam2 = data_syn.read_cameratimestamp('B:/SRIP_2019/depth_position_test/CameraTimestamp_nissan_test3.csv')
    gps1, x1, y1 = data_syn.read_gpstimestamp('B:/SRIP_2019/depth_position_test/infiniti_gps_3.csv')
    gps2, x2, y2 = data_syn.read_gpstimestamp('B:/SRIP_2019/depth_position_test/nissan_gps_3.csv')
    new_sync_multiple_time = data_syn.sync_multiple_cams(cam2, cam1)
    print(new_sync_multiple_time)
    infiniti_pair = data_syn.sync_cam_gps_timestamp(cam1, gps1)
    nissan_pair = data_syn.sync_cam_gps_timestamp(cam2, gps2)
    x1_one_list = np.ones(len(x1))
    y1_one_list = np.ones(len(y1))
    abs_x1pos_list = x1 - x1_one_list * x1[0]
    abs_y1pos_list = y1 - y1_one_list * y1[0]
    x2_one_list = np.ones(len(x2))
    y2_one_list = np.ones(len(y2))
    abs_x2pos_list = x2 - x2_one_list * x2[0]
    abs_y2pos_list = y2 - y2_one_list * y2[0]
    # infiniti_gps_relative_nissan_x = np.array(x1) - np.array(x2)
    # infiniti_gps_relative_nissan_y = np.array(y1) - np.array(y2)
    # print(infiniti_gps_relative_nissan_x)
    # print(infiniti_gps_relative_nissan_y)

    data_range = list(range(1797, 1843)) # depth img name num sequencial
    nissan_abs_gps_xlist = []
    nissan_abs_gps_ylist = []
    rel_inf_gps_position_xlist = []
    rel_inf_gps_position_ylist = []
    world_coor_xlist = []
    world_coor_ylist = []
    for i in data_range:
        k = str(i)
        while len(k) != 6:
            k = '0' + str(k)
        argument['img_name'] = k + 'depth.png'
        print(argument['img_name'])
        x, y = find_sample_data(argument['nissan_yolooutput_dir'], k, argument['number_features'])

        img = cv2.imread(argument['nissan_img_dir'] + argument['img_name'], cv2.IMREAD_UNCHANGED)

        median_depth, center_pixel_coor_list = depth_info_from_img(img, x)

    # print(median_depth)
        print(center_pixel_coor_list)

    # Used for test image Find positon
    # img_test = cv2.imread('B:/SRIP_2019/depth_position_test/test_img/000000depth.png', cv2.cv2.IMREAD_UNCHANGED)
    # test_pixel_coor = image_test(img_test)
    # optical_coor = pixel_to_optical(test_pixel_coor, intrinsic_mat_inv_test)
    # current_pose = np.array([0, 0, 0])
    # depth_camera_pos_ori = np.array([0, 0, 0])
    # world_coor = optical_to_world(optical_coor, current_pose, depth_camera_pos_ori)
    # print(world_coor)
    #     object_num = len(center_pixel_coor_list)
        object_num = 1

        for j in range(object_num):
            print(j)
            optical_coor = pixel_to_optical(center_pixel_coor_list[j], intrinsic_mat_inv)
            corre_img = new_sync_multiple_time[i]
            infiniti_gps_index = infiniti_pair[corre_img]
            nissan_gps_index = nissan_pair[i]

            infiniti_gps = [x1[infiniti_gps_index], y1[infiniti_gps_index]]
            nissan_gps = [x2[nissan_gps_index], y2[nissan_gps_index]]
            rel_inf_gps_position_x = infiniti_gps[0] - nissan_gps[0]
            rel_inf_gps_position_y = infiniti_gps[1] - nissan_gps[1]
            rel_inf_gps_position_xlist.append(x1[infiniti_gps_index])
            rel_inf_gps_position_ylist.append(y1[infiniti_gps_index])
            nissan_abs_gps = [x2[nissan_gps_index], y2[nissan_gps_index]]

            if nissan_abs_gps[0] not in nissan_abs_gps_xlist:

                nissan_abs_gps_xlist.append(nissan_abs_gps[0])
                nissan_abs_gps_ylist.append(nissan_abs_gps[1])

        # Set Nissan GPS module in [0, 0, 0] point as the Absolute Frame
                nissan_gps = np.array([0, 0, 0])
                world_coor = optical_to_world(optical_coor, nissan_gps, depth_camera_pos_ori)
                print(nissan_abs_gps)
                print(world_coor)
                world_coor = local_car_frame_to_earth_frame(world_coor, -117.234, 32.88)
    # print(infiniti_gps)
    # print(nissan_gps)
                print(world_coor)
                world_coor[0] = world_coor[0] + nissan_abs_gps[0]
                world_coor[1] = world_coor[1] + nissan_abs_gps[1]
                world_coor_xlist.append(world_coor[0])
                world_coor_ylist.append(world_coor[1])

                # fig = plt.figure()
                # plt.scatter(nissan_abs_gps_xlist, nissan_abs_gps_ylist, s=5, alpha=1.0, label='Nissan GPS Position')
                # plt.scatter(rel_inf_gps_position_xlist, rel_inf_gps_position_ylist, s=5, alpha=1.0,
                #             label='Infinitti GPS Position')
                # plt.scatter(world_coor_xlist, world_coor_ylist, s=5, alpha=1.0, label='Objects GPS Position')
                # # plt.scatter(nissan_abs_gps_xlist[0], nissan_abs_gps_xlist[0], s=50, color = 'red', label = 'Start')
                # # plt.scatter(nissan_abs_gps_ylist[-1], nissan_abs_gps_ylist[-1], s=50, color = 'green', label = 'End')
                # plt.xlabel('Absolute X Position unit(m)')
                # plt.ylabel('Absolute Y Position unit(m)')
                # plt.title('X and Y Position Value from GPS Module')
                # plt.legend(loc=1)
                # plt.draw()
                # plt.waitforbuttonpress(0)  # this will wait for indefinite time
                # plt.show()

    # x_diff = infiniti_gps[0] - x2[nissan_gps_index]
    # y_diff = infiniti_gps[1] - y2[nissan_gps_index]
    fig = plt.figure()
    plt.scatter(nissan_abs_gps_xlist, nissan_abs_gps_ylist, s=5, alpha=1.0, label = 'Nissan GPS Position')
    plt.scatter(rel_inf_gps_position_xlist, rel_inf_gps_position_ylist, s=5, alpha=1.0, label='Infinitti GPS Position')
    plt.scatter(world_coor_xlist, world_coor_ylist, s=5, alpha=1.0, label='Objects GPS Position')
    # plt.scatter(nissan_abs_gps_xlist[0], nissan_abs_gps_xlist[0], s=50, color = 'red', label = 'Start')
    # plt.scatter(nissan_abs_gps_ylist[-1], nissan_abs_gps_ylist[-1], s=50, color = 'green', label = 'End')
    plt.xlabel('Absolute X Position unit(m)')
    plt.ylabel('Absolute Y Position unit(m)')
    plt.title('X and Y Position Value from GPS Module')
    plt.legend(loc=1)
    plt.draw()
    plt.waitforbuttonpress(0) # this will wait for indefinite time
    plt.show()
    # print(x_diff)
    # print(y_diff)





if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--directory", type=str, help="Path to save the images")
    # parser.add_argument("-i", "--input", type=str, help="Bag file to read")
    # args = parser.parse_args()
    argument = {}
    # argument['test_img_name'] = 1756


    # argument['directory'] = 'B:\SRIP_2019\Image_test2_left'
    # argument['baginput'] = 'camera_calibrate_test/left15.bag'
    argument['directory'] = 'B:/SRIP_2019/depth_position_test'
    argument['number_features'] = 8
    argument['depth_max_range_value'] = 65.535
    argument['infiniti_yolooutput_dir'] =  argument['directory'] + '/test_infiniti/results.txt'
    argument['nissan_yolooutput_dir'] = argument['directory'] + '/test_nissan/position_test/results6_nissan.txt'
    argument['infiniti_img_dir'] = argument['directory'] + '/test_infiniti/depth/'
    argument['nissan_img_dir'] = argument['directory'] + '/test_nissan/position_test/depth/'


    # argument['baginput'] = 'camera_calibrate_test/right15.bag'
    # argument['csvfile'] = 'CameraTimestamp_second_stationary.csv'

    print(argument)

    # intrinsic_mat = np.array([[610.2642, 0, 313.8946],
    #                           [0, 610.9319, 235.1283],
    #                           [0, 0, 1]])

    intrinsic_mat = np.array([[601.753, 0, 311.193],
                              [0, 601.753, 238.364],
                              [0, 0, 1]])

    # intrinsic_mat_test = np.array([[599.419, 0, 315.129],
    #                           [0, 599.419, 238.771],
    #                           [0, 0, 1]])

    intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
    # intrinsic_mat_inv_test = np.linalg.inv(intrinsic_mat_test)
    # focal_length = 1.88e-3
    # intrinsic_mat = np.array([[-focal_length, 0, 0], [0, -focal_length, 0], [0, 0, 1]])
    # intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)

    main()


    # unit inches
    # x_offset = 36
    # y_offset = 10
    # z_offset = 27