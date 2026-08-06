from torch.utils.data import DataLoader
import torch.optim as optim
from model import Model
# from model_only_forward import Model

from dataset_test import Dataset
from test_10crop_test import test
from test_10crop_perf import test

# from test_10crop import test
import option
from utils import *
from config import *

# viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    seed_everything(args.seed)
    text_opt = "text_agg" if args.aggregate_text else "no_text_agg"
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    viz_name = 'share_feedback_mode-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(args.dataset, args.feat_extractor, args.feature_group,
                                                            text_opt,  args.emb_folder, args.fusion, args.decoders_mode, args.feedback_mode, args.omega_v,
                                                            args.omega_t, args.margin_prompt, args._lambda,
                                                            args.feedback_aug, args.max_epoch, args.seed, args.lr, args.batch_size, args.prompt_num)

    model = Model(args)
    if args.pretrained_ckpt is not None:
        print("Loading pretrained model " + args.pretrained_ckpt)
        model.load_state_dict(torch.load(args.pretrained_ckpt))
    else:
        if "shanghai" in args.dataset:
            model.load_state_dict(torch.load('./ckpt/my_best/shanghai_v2-both-text_agg-add-1-1-extra_loss-595-i3d-best.pkl'))
        elif "ped2" in args.dataset:
            model.load_state_dict(torch.load('./ckpt/my_best/ped2-both-text_agg-add-1-1-extra_loss-755-4869-i3d.pkl'))
        elif "violence" in args.dataset:
            model.load_state_dict(torch.load('./ckpt/my_best/violence-both-text_agg-add-1-1-extra_loss-445-4869-BEST.pkl'))
        elif "ucf" in args.dataset:
            model.load_state_dict(torch.load('./ckpt/my_best/ucf-both-text_agg-concat-1-1-extra_loss-680-2333-BEST.pkl'))
        elif "TE2" in args.dataset:  # ped2 model works better
            model.load_state_dict(
                torch.load('./ckpt/my_best/TE2-both-text_agg-concat-0.0001-extra_loss-645-4869-.pkl'))
        else:
            raise NotImplementedError

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    best_AUC = -1
    output_path = 'output_test'   # put your own path here
    # rec_auc_all, ap, rec_auc_abn, far_all, far_abn = test(test_loader, model, args, viz, device, plot_curve=True)
    # rec_auc_all, ap, rec_auc_abn, far_all, far_abn = test(test_loader, model, args, viz, viz_name, device, best_AUC, plot_curve=True)
    rec_auc_all, ap, rec_auc_abn, far_all, far_abn = test(test_loader, model, args, viz_name, device, best_AUC, plot_curve=True)

