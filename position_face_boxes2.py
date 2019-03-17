#！/home/w/soft/anaconda2/bin/python

#!/usr/bin/python
# #encoding=utf-8
import pickle
from multiprocessing import Pool, Queue, Manager
import subprocess
import os.path as osp
import os
import cv2
import numpy as np
import pandas as pd
import time
import argparse
from test import long_time_task
import datetime
import gc
import copy


def parse_arguments(args):
    parse = argparse.ArgumentParser(description="提取iqiyi官方给的框")
    parse.add_argument("--feats_dict_path", type=str, help="官方提供的feat_dict的路径", required=True)
    parse.add_argument("--py2", action="store_false", dest="does_feats_dict_py3",
                       help="提供的feats_dict是否是python2格式的, 默认是python3的")
    parse.add_argument("--video_dir_path", type=str, help="视频文件的路径, 所有对应于--txt_file_path的视频都在这里面", required=True)
    parse.add_argument("--video_dir_path2", type=str, help="视频文件的路径, 所有对应于--txt_file_path的视频都在这里面", required=True)
    parse.add_argument("--video_dir_path3", type=str, help="视频文件的路径, 所有对应于--txt_file_path的视频都在这里面", required=True)
    parse.add_argument("--txt_file_path", type=str, help="官方提供的那个txt文件的位置", required=True)
    parse.add_argument("--output_dir_path", type=str, help="这个文件夹下存放着所有本程序的输出, 包括从视频临时提取出来的图片文件夹, 人脸图片文件夹, meta文件夹",
                       required=True)
    #
    # parse.add_argument("--output_picture_dir_path", type=str, help="根据官方给的框而提出来的人脸的图像应该放在什么位置", required=True)
    # parse.add_argument("--tmp_pic_dir_path", type=str, help="由于中间需要根据ffmp截取视频,这个视频图片应该放在按个文件夹下", required=True)
    # parse.add_argument("--meta_dir_path", type=str, help="存放没有人脸的视频的记录的地址, 也保存官方给的那个dict的pandas版本")

    parse.add_argument("--how_many_process", type=int, help="启用多少个进程进行处理", default=2)
    parse.add_argument("--face_size", type=int, help="提取出来的人脸的大小, 会resize成这个大小", default=160)
    parse.add_argument("--min_det", type=float, help="det 大于这个的留下,其他的不留下, 需要和min_qua一起满足", default=0.97)
    parse.add_argument("--min_qua", type=float, help="qua 大于这个的留下,其他的不留下, 需要和min_det一起满足", default=30)
    parse.add_argument("--min_num", type=int, help="qua最高的几个一定会被存下来", default=3)
    parse.set_defaults(does_feats_dict_py3=True)

    return parse.parse_args(args)


def sub_job(people_idx, feats_dict_value_by_video_name, video_name,tmp_video_path,
            no_face_viedo_names_queue: Queue, output_picture_dir_path,output_picture_dir_path_with_people_idx,dataframe_queue: Queue,face_size=160,
            index="dont_care", min_det=0.97, min_qua=40, min_num=3
            ):
    t1 = time.time()

    #can't meet
    det_less_dir_path = osp.join(
        "{}_det_less_than_{}_or_qua_less_than_{}_exclude_one_video_has_less_than_{}_face".format(
            output_picture_dir_path, min_det, min_qua, min_num), people_idx)

    image_names = list(filter(lambda x: osp.isfile(osp.join(tmp_video_path, x)), os.listdir(tmp_video_path)))  #name list
    raw_image_paths = list(map(lambda x: osp.join(tmp_video_path, x), image_names))# path  list

    tmp_df = pd.DataFrame(
        feats_dict_value_by_video_name,
        columns=['frame_id', 'box', 'det_score', 'qua_score', 'feat_arr']
    ).assign(
        video_name=video_name
    ).set_index("video_name", drop=True).assign(people_id=people_idx).sort_values("qua_score", ascending=False)


    #dataframe_queue.put(tmp_df)

    if len(tmp_df) == 0:
        no_face_viedo_names_queue.put(video_name)
        print(
            "people {}({}) done, this processing cost {} seconds, index is {}, this video has no face".format(
                people_idx, video_name, time.time() - t1, index))
        return#No face ,so return
    # 这一行放在下面,因为有可能长度为0, 导致不能iloc
    min_qua_self = np.mean(tmp_df.iloc[min_num - 1]['qua_score']) if len(tmp_df)>=min_num else -1#   3 mean

    for image_path, image_name in zip(raw_image_paths, image_names):

        frame_id = int(image_name.split(".")[0]) #000001.jpg---000001
        (boxes, dets, quas) = tmp_df[tmp_df.frame_id == frame_id].pipe(
            lambda x: x.sort_values("qua_score", ascending=False)).pipe(
            lambda x: (x.box.tolist(), x.det_score.tolist(), x.qua_score.tolist())
        )#? boxes?
        tmp_df = tmp_df[tmp_df.frame_id != frame_id]#?

        assert len(boxes) == len(dets) == len(quas)

        if len(boxes) == 0:
            continue

        img = cv2.imread(image_path)
        img_size = np.asarray(img.shape)[0:2]#  list? 896*504*3
        for box, det, qua, i in zip(boxes, dets, quas, list(range(len(boxes)))):
            output_picture_path = osp.join(output_picture_dir_path_with_people_idx,
                                           "{}_{}_{}_{}_{}.jpg".format(video_name, frame_id, i, det, qua))

            det_less_path = osp.join(det_less_dir_path,
                                     "{}_{}_{}_{}_{}.jpg".format(video_name, frame_id, i, det, qua))

            x1 = np.maximum(int(box[0]) - 16, 0)
            y1 = np.maximum(int(box[1]) - 16, 0)
            x2 = np.minimum(int(box[2]) + 16, img_size[1])
            y2 = np.minimum(int(box[3]) + 16, img_size[0]) #  ?
            crop_img = img[y1:y2, x1:x2]
            scaled = cv2.resize(crop_img, (face_size, face_size))

            if (det >= min_det and float(qua) >= min_qua) or float(qua)>=min_qua_self:
                output_picture_path_or_det_less_path = output_picture_path
                #cv2.imwrite(output_picture_path, scaled)
            else:
                output_picture_path_or_det_less_path = det_less_path
                while True:
                    try:
                        if not osp.exists(det_less_dir_path):
                            os.makedirs(det_less_dir_path)
                    except:
                        continue
                    break
            while True:
                try:
                    cv2.imwrite(output_picture_path_or_det_less_path, scaled)
                except:
                    continue
                break

    print("people {}({}) done, this processing cost {} seconds, index is {}".format(people_idx, video_name,
                                                                                    time.time() - t1, index))
    del tmp_df


def main(feats_dict_path, txt_file_path, video_dir_path,video_dir_path2,video_dir_path3,output_picture_dir_path, tmp_pic_dir_path, meta_dir_path,
         how_many_process: int = 15, does_feats_dict_py3=True, face_size=160, min_det=0.97, min_qua=30, min_num=3):
    manager = Manager()
    no_face_video_names_queue = manager.Queue()
    dataframe_queue = manager.Queue()
    if not osp.exists(meta_dir_path):
        os.makedirs(meta_dir_path)
    if not osp.exists(output_picture_dir_path):
        os.makedirs(output_picture_dir_path)
    if not osp.exists(tmp_pic_dir_path):
        os.makedirs(tmp_pic_dir_path)

    people_dict = {}

    with open(txt_file_path, 'r') as r:
        for line in r:
            filename, idx = line.split()
            if idx not in people_dict:   #将已有名字的的视频整合到一个id下
                people_dict[idx] = []
            people_dict[idx].append(filename)
    # print(people_dict[101])
    t1 = time.time()
    with open(feats_dict_path, 'rb') as f:
        # 官方提供的pickle应该再加上encoding, 具体看https://blog.csdn.net/accumulateargparse.ArgumentParser_zhang/article/details/78597823
        if does_feats_dict_py3:
            feats_dict = pickle.load(f)
        else:
            feats_dict = pickle.load(f, encoding="iso-8859-1")

    t2 = time.time()
    print("load feat dict cost {} seconds".format(t2 - t1))

    tmp_list = []
    #第一步: 提取帧
    for people_idx, video_names in people_dict.items():
        for video_name in video_names:
            if video_name not in feats_dict:
                continue
            #将这个视频与官方给的三个视频集正确匹配，确定视频路径
            video_path = osp.join(video_dir_path, video_name)
            if not os.path.isfile(video_path):
                video_path = osp.join(video_dir_path2, video_name)
                if not os.path.isfile(video_path):
                   video_path = osp.join(video_dir_path3, video_name)
            #提取帧的存放路径
            tmp_video_path = osp.join(tmp_pic_dir_path, people_idx, video_name)

            # if osp.exists(tmp_video_path):   什么作用
            #     continue
            if not osp.exists(tmp_video_path):
                os.makedirs(tmp_video_path)

            #提取帧操作
            subprocess.run(
                    ["ffmpeg -i {} -vf fps=fps=8/1 -q 0 {}".format(video_path, osp.join(tmp_video_path, "%06d.jpg"))],
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            tmp_df = pd.DataFrame(
                feats_dict[video_name],
                columns=['frame_id', 'box', 'det_score', 'qua_score', 'feat_arr']
            ).assign(
                video_name=video_name
            ).set_index("video_name", drop=True).assign(people_id=people_idx)
            tmp_list.append(tmp_df)
            # del feats_dict[video_name] # 平时运行一定要注释掉
    pd.concat(tmp_list).to_pickle(osp.join(meta_dir_path, "feats_df.pickle"))#提取帧的特征存为pickle？与官方给的pickle区别在哪？
    print("saving features done! cost {} seconds".format(time.time() - t1))
    del tmp_df, tmp_list



    # 第二步: 提取人脸
    pool = Pool(how_many_process)
    i = 0
    for people_idx, video_names in people_dict.items():
        collected = gc.collect()
        print("clear {}".format(collected))
        for video_name in video_names:
            if video_name not in feats_dict:
                continue

            tmp_video_path = osp.join(tmp_pic_dir_path, people_idx, video_name)
            #提取人脸的存放路径
            output_picture_dir_path_with_people_idx = osp.join(output_picture_dir_path, people_idx)
            if not osp.exists(output_picture_dir_path_with_people_idx):
                os.makedirs(output_picture_dir_path_with_people_idx)

            tmp = copy.deepcopy(feats_dict.pop(video_name))
            args = (
                people_idx, tmp, video_name, tmp_video_path,
                no_face_video_names_queue,output_picture_dir_path,output_picture_dir_path_with_people_idx,dataframe_queue
            )
            kwds = {"face_size": face_size, "index": i, "min_det": min_det, "min_qua": min_qua, "min_num": min_num}

            # debug
            # sub_job(*args, **kwds)
            # debug done

            pool.apply_async(sub_job, args=args, kwds=kwds)
            i += 1
            # pool.apply_async(long_time_task, args=("{}_{}".format(people_idx, video_name)))
            del args, tmp
            print("garbage are: {}".format(gc.garbage))

    pool.close()
    pool.join()
    print(
        "done!!, total cost {} seconds and process video cost {} seconds, per-person cost {} seconds(average)".format(
            time.time() - t1, time.time() - t2, (time.time() - t2) / len(people_dict)
        ))

    with open(osp.join(meta_dir_path, "no_face_video_names.txt"), 'a') as f:
        while not no_face_video_names_queue.empty():
            f.write(no_face_video_names_queue.get(block=True) + "\n")

    print("current time is {}".format(datetime.datetime.now()))

import sys
parse = parse_arguments(sys.argv[1:])

main(parse.feats_dict_path, parse.txt_file_path, parse.video_dir_path, parse.output_picture_dir_path,
     parse.tmp_pic_dir_path, parse.meta_dir_path, parse.how_many_process, parse.does_feats_dict_py3, parse.face_size)


if __name__ == "__main__":
    import sys

    parse = parse_arguments(sys.argv[1:])

    output_feace_dir_path = osp.join(parse.output_dir_path, "face_location")#parse.output_dir_path=/home/w/competition/aiqiyi/data/IQIYI_VID_DATA_Part1_position_box_by_official, there should be three ?
    output_tmp_dir_path = osp.join(parse.output_dir_path, "tmp_pic_extract_from_video")
    output_meta_dir_path = osp.join(parse.output_dir_path, "meta_datas")

    main(
        parse.feats_dict_path, parse.txt_file_path, parse.video_dir_path,parse.video_dir_path2,parse.video_dir_path3,output_feace_dir_path,
        output_tmp_dir_path, output_meta_dir_path,
        parse.how_many_process, parse.does_feats_dict_py3, parse.face_size, parse.min_det,
        parse.min_qua, parse.min_num
    )
