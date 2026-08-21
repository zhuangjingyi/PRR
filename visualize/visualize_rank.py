from __future__ import print_function, absolute_import
import numpy as np
import os
import shutil


def visualize_vessel(distmat, q_pids, g_pids, query_fnames, gallery_fnames, output_dir='./vis_results/'):
    max_rank = 20
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)

    indices_idxs = indices


    for q_id in range(len(indices_idxs)):       #迭代查询图像的数量
        for g_id in range(20):
            q_pid = q_pids[q_id]
            g_pid = g_pids[indices_idxs[q_id][g_id]]
            q_img_oldpath = query_fnames[q_id]
            g_img_oldpath = gallery_fnames[indices_idxs[q_id][g_id]]

            output_rank_info = os.path.join(output_dir, "rank_vessel_proposed", str(q_id))
            if not os.path.exists(output_rank_info):
                os.makedirs(output_rank_info)

            if q_pid == g_pid:
                is_match = "_match_"
            else:
                is_match = "_mismatch_"



            q_img = q_img_oldpath[:q_img_oldpath.rfind(".")][q_img_oldpath.rfind('/')+1:]
            g_img = g_img_oldpath[:g_img_oldpath.rfind(".")][g_img_oldpath.rfind('/')+1:]


            q_img_new_name = "q_" + q_img
            g_img_new_name = "g_" + str(g_id) + is_match + g_img


            q_dist_path = os.path.join(output_rank_info, q_img_new_name)
            q_dist_path = q_dist_path + '.jpg'
            g_dist_path = os.path.join(output_rank_info, g_img_new_name)
            g_dist_path = g_dist_path + '.jpg'
            shutil.copy(q_img_oldpath, q_dist_path)
            shutil.copy(g_img_oldpath, g_dist_path)
