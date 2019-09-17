import csv
import datetime
import time
import numpy as np
import os

hours_modify = 7
def read_cameratimestamp(filename):
    with open(filename) as csvfile1:
        readCSV_cam = csv.reader(csvfile1, delimiter=',')
        cam_index_list = []
        cam_time_list = []
        for row in readCSV_cam:
            if row != []:
                cur_index = int(row[0])
                cur_cam_time = float(row[1])/1000
                if cur_cam_time not in cam_time_list:
                    cam_index_list.append(cur_index)
                    cam_time_list.append(cur_cam_time)
    return cam_time_list

def read_gpstimestamp(filename):
    with open(filename) as csvfile2:
        readCSV_gps = csv.reader(csvfile2, delimiter=',')
        gps_index_list = []
        gps_utctime_list = []
        gps_sattime_list = []
        # gps_sog_list = []
        # gps_cog_list = []
        gps_x_list = []
        gps_y_list = []
        # gps_z_list = []
        # gps_lon_list = []
        # gps_lat_list = []
        for row in readCSV_gps:
            if row != []:
                cur_index = int(row[0])
                cur_gps_utctime = row[1]
                cur_gps_utctime_ms = int(cur_gps_utctime[9:12]) / 1000
                cur_gps_utc_unix = time.mktime(datetime.datetime.strptime(cur_gps_utctime, '%H:%M:%S.%f %m/%d/%Y').timetuple()) + cur_gps_utctime_ms - hours_modify*3600
                cur_gps_sattime = row[2]
                # cur_gps_sog = float(row[3])
                # if row[4] != '':
                #     cur_gps_cog = float(row[4])
                if row[3] != '':
                    cur_gps_x = float(row[3])
                    cur_gps_y = float(row[4])
                # cur_gps_z = float(row[7])
                # cur_gps_lon = float(row[8])
                # cur_gps_lat = float(row[9])
                gps_index_list.append(cur_index)
                gps_utctime_list.append(cur_gps_utc_unix)
                gps_sattime_list.append(cur_gps_sattime)
                # gps_sog_list.append(cur_gps_sog)
                # gps_cog_list.append(cur_gps_cog)
                gps_x_list.append(cur_gps_x)
                gps_y_list.append(cur_gps_y)
                # gps_z_list.append(cur_gps_z)
                # gps_lon_list.append(cur_gps_lon)
                # gps_lat_list.append(cur_gps_lat)
    return gps_utctime_list, gps_x_list, gps_y_list

def sync_cam_gps_timestamp(cam_timestamp, gps_timestamp):
    new_pair_index_dict = {}
    for i in range(len(cam_timestamp)):
        one_list = np.ones(len(gps_timestamp))
        substract_list = np.abs(np.array(gps_timestamp) - cam_timestamp[i] * one_list)
        cur_min_value = min(substract_list)
        if cur_min_value < 1:
            cur_min_index = np.argmin(substract_list)
            # index_pair = [i, cur_min_index]
            # new_pair_index_list.append(index_pair)
            if i not in new_pair_index_dict:
                new_pair_index_dict[i] = cur_min_index
    return new_pair_index_dict

def sync_multiple_cams(gps1, gps2):
    new_pair_index_dict = sync_cam_gps_timestamp(gps1, gps2)
    return new_pair_index_dict

# def find_out_dist(desired_img_name, ):
def Euclidean_Dist(x, y):
    x1 = x[0]
    x2 = y[0]
    y1 = x[1]
    y2 = y[1]
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return dist

if __name__ == "__main__":

    desired_img_name = 945 # changed as the depth img num is different

    cam1 = read_cameratimestamp('B:/SRIP_2019/depth_position_test/CameraTimestamp_infiniti_img.csv')
    cam2 = read_cameratimestamp('B:/SRIP_2019/depth_position_test/CameraTimestamp_nissan_left.csv')
    gps1, x1, y1 = read_gpstimestamp('B:/SRIP_2019/depth_position_test/infiniti_gps_4.csv')
    gps2, x2, y2 = read_gpstimestamp('B:/SRIP_2019/depth_position_test/nissan_gps_4.csv')

    new_sync_multiple_time = sync_multiple_cams(cam2, cam1)
    infiniti_pair = sync_cam_gps_timestamp(cam1, gps1)
    nissan_pair = sync_cam_gps_timestamp(cam2, gps2)

    corre_img = new_sync_multiple_time[desired_img_name]
    infiniti_gps_index = infiniti_pair[corre_img]
    nissan_gps_index = nissan_pair[desired_img_name]

    infiniti_gps = [x1[infiniti_gps_index], y1[infiniti_gps_index]]
    nissan_gps = [x2[nissan_gps_index], y2[nissan_gps_index]]

    euclidean_dist = Euclidean_Dist(infiniti_gps, nissan_gps)
    print(infiniti_gps)
    print(nissan_gps)
    print(euclidean_dist)
    # for i in range(len(new_sync_multiple_time)):
    #     if new_sync_multiple_time[i][0] == desired_img_name:
    #         corre_img = new_sync_multiple_time[i][1]
    # for j in range(len(infiniti_pair)):
    #     if infiniti_pair[]

    # print(new_sync_multiple_time)
    # print(corre_img)
    # print(infiniti_pair)
    # print(nissan_pair)
    # print(len(x1))
    # print(len(y1))

# save_sog_path = 'sync_test/sog.npy'
# save_cog_path = 'sync_test/cog.npy'
# save_gps_timestamp_path = 'sync_test/gps_timestamp.npy'
# save_x_path = 'sync_test/x_pos.npy'
# save_y_path = 'sync_test/y_pos.npy'
# save_z_path = 'sync_test/z_pos.npy'
# save_lon_path = 'sync_test/lon_pos.npy'
# save_lat_path = 'sync_test/lat_pos.npy'
# save_sattime_path = 'sync_test/sat_time.npy'

# if not os.path.exists(save_sog_path):
#     np.save(save_sog_path, gps_sog_list)
#     print(1)
# if not os.path.exists(save_cog_path):
#     np.save(save_cog_path, gps_cog_list)
#     print(2)
# if not os.path.exists(save_gps_timestamp_path):
#     np.save(save_gps_timestamp_path, gps_utctime_list)
#     print(3)
# if not os.path.exists(save_x_path):
#     np.save(save_x_path, gps_x_list)
#     print(4)
# if not os.path.exists(save_y_path):
#     np.save(save_y_path, gps_y_list)
#     print(5)
# if not os.path.exists(save_sattime_path):
#     np.save(save_sattime_path, gps_sattime_list)
#     print(6)
# if not os.path.exists(save_z_path):
#     np.save(save_z_path, gps_z_list)
#     print(7)
# if not os.path.exists(save_lon_path):
#     np.save(save_lon_path, gps_lon_list)
#     print(8)
# if not os.path.exists(save_lat_path):
#     np.save(save_lat_path, gps_lat_list)
#     print(9)

# print(cam_time_list[new_pair_index_list[10][0]])
# print(gps_utctime_list[new_pair_index_list[10][1]])

# date_time_str = gps_utctime_list[104]
# print(int(date_time_str[9:12])/1000)
# date_time_obj = datetime.datetime.strptime(date_time_str, '%H:%M:%S.%f %m/%d/%Y')
# print(time.mktime(datetime.datetime.strptime(date_time_str, '%H:%M:%S.%f %m/%d/%Y').timetuple()) + int(date_time_str[9:12])/1000)
# print()
#
# print()