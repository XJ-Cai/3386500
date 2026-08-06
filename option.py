import argparse

parser = argparse.ArgumentParser(description='RTFM')
parser.add_argument('--feat_extractor', default='clip', choices=['i3d', 'c3d', 'videoMAE', 'clip'])
parser.add_argument('--feature-size', type=int, default=512, help='size of vis feature (default: 2048)')
# parser.add_argument('--feature_size', type=int, default=512, help='size of vis feature (default: 2048)')
parser.add_argument('--use_dic_gt', action='store_true', default=True,  help='get GrandTruth from a pickle file')
parser.add_argument('--alignment_method', type=str, default='add',choices=['add', 'cut'],  help='the alignment method to dealwith superfluous frames')
parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
parser.add_argument('--rgb_list', default='list/ucf-clip-train.list', help='list of rgb features ')
parser.add_argument('--test_rgb_list', default='list/ucf-clip-test.list', help='list of test rgb features ')
parser.add_argument('--gt', default=None, help='file of ground truth ')
parser.add_argument('--gpus', default=1, type=int, choices=[0], help='gpus')
parser.add_argument('--lr', type=str, default='[0.001]*15000', help='learning rates for steps(list form)')
parser.add_argument('--batch_size', type=int, default=32, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--model-name', default='rtfm', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--dataset', default='ucf', help='dataset to train on (shanghai, ucf, ped2, violence, TE2)')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')

parser.add_argument('--seed', type=int, default=228, help='random seed (default: 4869)')
parser.add_argument('--max_epoch', type=int, default=1000, help='maximum iteration to train (default: 1000)')
parser.add_argument('--feature-group', default='both', choices=['both', 'vis', 'text'], help='feature groups used for the model')
parser.add_argument('--fusion', type=str, default='add', help='how to fuse vis and text features')
parser.add_argument('--normal_weight', type=float, default=1, help='weight for normal loss weights')
parser.add_argument('--abnormal_weight', type=float, default=1, help='weight for abnormal loss weights')
parser.add_argument('--aggregate_text', action='store_true', default=True, help='whether to aggregate text features')
parser.add_argument('--extra_loss', action='store_true', default=True, help='whether to use extra loss')
parser.add_argument('--save_test_results', action='store_true', default=False, help='whether to save test results')
parser.add_argument('--alpha', type=float, default=0.0001, help='weight for RTFM loss')
parser.add_argument('--emb_folder', type=str, default='sent_emb_n', help='folder for text embeddings, used to differenciate different swinbert pretrained models')
parser.add_argument('--emb_dim', type=int, default=768, help='dimension of text embeddings')

parser.add_argument('--abn_curve_save_root', type=str, default='./figures', help='folder for abn_curve_savepath')
parser.add_argument('--beta', type=float, default=0.0, help='weight for LAT loss')
parser.add_argument('--gama', type=float, default=0.0, help='weight for triplet loss')

# decoder mode
parser.add_argument('--decoders_mode', type=str, default='geu', choices=['mlp', 'geu', 'down_up'])
parser.add_argument('--feedback_mode', type=str, default='reweight', choices=['reweight', 'concat', 'add', 'product'])
parser.add_argument('--depth', type=int, default=2)

# prompt
parser.add_argument('--_lambda', type=float, default=0.01, help='weight for prompt margin loss')
parser.add_argument('--margin_prompt', type=float, default=30, help='weight for prompt margin loss')
parser.add_argument('--feedback_aug', type=int, default=20, help='weight for prompt margin loss')
parser.add_argument('--prompt_num', type=int, default=2, help='(2+2)')
parser.add_argument('--p_normal', type=int, default=0, help='')

# var_loss set
parser.add_argument('--omega_v', type=float, default=0.8, help='weight for var_loss_visual')
parser.add_argument('--omega_t', type=float, default=0.01, help='weight for var_loss_text')

# parser.add_argument('--theta', type=float, default=0.5, help='weight for similarity')





