import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("--vis", action="store_true", help="Enable rank visualization")
    parser.add_argument("--tsne", action="store_true", help="Enable t-SNE feature visualization")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # ---- 根据超参数推导「权重所在目录」（必须和 train.py 完全一致，不含 eval_mode） ----
    s_pos = cfg.SOLVER.HARD_EXAMPLE_POS_K
    s_neg = cfg.SOLVER.HARD_EXAMPLE_NEG_K
    tri_weight = cfg.MODEL.TRIPLET_LOSS_WEIGHT
    interaction_interval = cfg.MODEL.INTERACTION_INTERVAL
    loss_type = cfg.MODEL.METRIC_LOSS_TYPE
    # auto_dir_name = "pos{}_neg{}_Frequency{}".format(s_pos, s_neg, interaction_interval)
    # auto_dir_name = "pos{}_neg{}_Frequency{}_Tri{}".format(s_pos, s_neg,interaction_interval,tri_weight)
    auto_dir_name = "pos{}_neg{}_Frequency{}_Tri{}_LossType_{}".format(s_pos, s_neg,interaction_interval,tri_weight,loss_type)
    ckpt_dir = os.path.join(cfg.OUTPUT_DIR, auto_dir_name)   # ← 权重目录，和 train.py 保持一致

    # ---- 日志目录：在权重目录基础上，NightReID 额外加 eval_mode 子目录用于区分测试配置 ----
    if cfg.DATASETS.NAMES == 'nightreid':
        eval_mode = cfg.DATASETS.EVAL_MODE
        log_dir = os.path.join(ckpt_dir, "test_{}".format(eval_mode))
    else:
        log_dir = ckpt_dir

    cfg.OUTPUT_DIR = log_dir   # 日志存到这里

    test_epoch = cfg.TEST.TEST_EPOCHS
    if "TEST.WEIGHT" not in args.opts:
        ckpt_name = "{}_{}.pth".format(cfg.MODEL.NAME, test_epoch)
        # ckpt_name = "{}_best_mAP.pth".format(cfg.MODEL.NAME)
        cfg.TEST.WEIGHT = os.path.join(ckpt_dir, ckpt_name)   # 权重从 ckpt_dir 找，不带 eval_mode 后缀
    # -----------------------------
    print("---------weight:",cfg.TEST.WEIGHT)
    log_name = f"test_{test_epoch}_log.txt"

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False, log_name=log_name)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)

    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5 = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))
    else:
       do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 vis_rank=args.vis,
                 tsne=args.tsne)