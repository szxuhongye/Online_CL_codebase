import argparse
import random
import numpy as np
import torch
from experiment.run import multiple_run
from utils.utils import boolean_string
import logging
import sys
from utils.name_match import agents
from utils.setup_elements import setup_opt, setup_architecture
from utils.utils import maybe_cuda

class DualLogger:
    """
    A custom logger that logs to both the console and a file.
    """

    def __init__(self, name, log_file, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Prevent logging twice

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create file handler, set its level and formatter
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        # Create stream handler (console), set its level and formatter
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)

        # Add both handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def write(self, message):
        if message.strip() != "":
            self.logger.info(message.strip())

    def flush(self):
        # This could be fleshed out to actually flush the handlers
        pass

    def get_logger(self):
        """
        Returns the logger instance.
        """
        return self.logger

def add_agent_parse(initial_args, parser):
    initial_args.cuda = torch.cuda.is_available()
    model = setup_architecture(initial_args)
    model = maybe_cuda(model, initial_args.cuda)
    opt = setup_opt(initial_args.optimizer, model, initial_args.learning_rate, initial_args.weight_decay)
    agent = agents[initial_args.agent](model, opt, initial_args)
    ## check whether agent have attribute add_agent_args
    if hasattr(agent, 'add_agent_args'):
        agent.add_agent_args(parser)
    


def main(args):
    print(args)
    # set up seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    args.trick = {'labels_trick': args.labels_trick, 'separated_softmax': args.separated_softmax,
                  'kd_trick': args.kd_trick, 'kd_trick_star': args.kd_trick_star, 'review_trick': args.review_trick,
                  'ncm_trick': args.ncm_trick}
    
    log_file = '{}.log'.format(args.exp)
    dual_logger = DualLogger(__name__, log_file)
    # Redirect standard print statements to the logger
    # Shengjie
    # sys.stdout = dual_logger

    multiple_run(args, store=args.store, save_path=args.save_path)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Online Continual Learning PyTorch")
    ########################General#########################
    parser.add_argument('--num_runs', dest='num_runs', default=1, type=int,
                        help='Number of runs (default: %(default)s)')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='Random seed')

    ########################Misc#########################
    parser.add_argument('--val_size', dest='val_size', default=0.1, type=float,
                        help='val_size (default: %(default)s)')
    parser.add_argument('--num_val', dest='num_val', default=3, type=int,
                        help='Number of batches used for validation (default: %(default)s)')
    parser.add_argument('--num_runs_val', dest='num_runs_val', default=3, type=int,
                        help='Number of runs for validation (default: %(default)s)')
    parser.add_argument('--error_analysis', dest='error_analysis', default=False, type=boolean_string,
                        help='Perform error analysis (default: %(default)s)')
    parser.add_argument('--verbose', type=boolean_string, default=True,
                        help='print information or not (default: %(default)s)')
    parser.add_argument('--store', type=boolean_string, default=False,
                        help='Store result or not (default: %(default)s)')
    parser.add_argument('--save-path', dest='save_path', default=None)
    parser.add_argument('--imagenet-path', dest='imagenet_path', default='./imagenet1k')

    ########################Agent#########f################
    parser.add_argument('--agent', dest='agent', default='PCR',
                        choices=['PCR', 'CLSER', 'ER', 'ER_ACE', 'ER_ACE_L', 'EWC', 'AGEM', 'CNDPM', 'LWF', 'ICARL', 'GDUMB', 'ASER', 'SCR', 'SUPER', 'SuperPCR', 'GEM', 'DERPP'],
                        help='Agent selection  (default: %(default)s)')
    parser.add_argument('--update', dest='update', default='random', choices=['random', 'GSS', 'ASER'],
                        help='Update method  (default: %(default)s)')
    parser.add_argument('--retrieve', dest='retrieve', default='random', choices=['MIR', 'random', 'ASER', 'match', 'mem_match', 'UCR', 'HDR', 'IMIR', 'MIX'],
                        help='Retrieve method  (default: %(default)s)')
    
    parser.add_argument('--second_buffer', dest='second_buffer', default=False,
                        type=boolean_string,
                        help='If False, no second buffer will be used (default: %(default)s)')
    parser.add_argument('--update2', dest='update2', default='random', choices=['random', 'GSS', 'ASER', 'IGSS'],
                        help='Second Buffer Update method  (default: %(default)s)')
    parser.add_argument('--retrieve2', dest='retrieve2', default='random', choices=['MIR', 'random', 'ASER', 'match', 'mem_match', 'IMIR', 'HDR', 'MIX'],
                        help='Second Buffer Retrieve method  (default: %(default)s)')
    # parser.add_argument('--ratio', dest='ratio', default=0.2,
    #                     type=float,
    #                     help='Learning_rate (default: %(default)s)')

    ########################Optimizer#########################
    parser.add_argument('--optimizer', dest='optimizer', default='SGD', choices=['SGD', 'Adam'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.1,
                        type=float,
                        help='Learning_rate (default: %(default)s)')
    parser.add_argument('--epoch', dest='epoch', default=50,
                        type=int,
                        help='The number of epochs used for one task. (default: %(default)s)')
    parser.add_argument('--batch', dest='batch', default=256,
                        type=int,
                        help='Batch size (default: %(default)s)')
    parser.add_argument('--test_batch', dest='test_batch', default=128,
                        type=int,
                        help='Test batch size (default: %(default)s)')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0,
                        help='weight_decay')

    ########################Data#########################
    parser.add_argument('--num_tasks', dest='num_tasks', default=20,
                        type=int,
                        help='Number of tasks (default: %(default)s), OpenLORIS num_tasks is predefined')
    parser.add_argument('--fix_order', dest='fix_order', default=False,
                        type=boolean_string,
                        help='In NC scenario, should the class order be fixed (default: %(default)s)')
    parser.add_argument('--plot_sample', dest='plot_sample', default=False,
                        type=boolean_string,
                        help='In NI scenario, should sample images be plotted (default: %(default)s)')
    parser.add_argument('--data', dest='data', default="cifar100",
                        help='Path to the dataset. (default: %(default)s)')
    parser.add_argument('--cl_type', dest='cl_type', default="nc", choices=['nc', 'ni'],
                        help='Continual learning type: new class "nc" or new instance "ni". (default: %(default)s)')
    parser.add_argument('--ns_factor', dest='ns_factor', nargs='+',
                        default=(0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6), type=float,
                        help='Change factor for non-stationary data(default: %(default)s)')
    parser.add_argument('--ns_type', dest='ns_type', default='noise', type=str, choices=['noise', 'occlusion', 'blur'],
                        help='Type of non-stationary (default: %(default)s)')
    parser.add_argument('--ns_task', dest='ns_task', nargs='+', default=(1, 1, 2, 2, 2, 2), type=int,
                        help='NI Non Stationary task composition (default: %(default)s)')
    parser.add_argument('--online', dest='online', default=True,
                        type=boolean_string,
                        help='If False, offline training will be performed (default: %(default)s)')
    parser.add_argument('--use_momentum', dest='use_momentum', default=True,
                        type=boolean_string,
                        help='If True, will use momemtum encoders(default: %(default)s)')

    ########################ER#########################
    parser.add_argument('--mem_size', dest='mem_size', default=1000,
                        type=int,
                        help='Memory buffer size (default: %(default)s)')
    parser.add_argument('--eps_mem_batch', dest='eps_mem_batch', default=10,
                        type=int,
                        help='Episode memory per batch (default: %(default)s)')
    parser.add_argument('--sub_eps_mem_batch', dest='sub_eps_mem_batch', default=256,
                        type=int,
                        help='Sub Episode memory per batch (default: %(default)s)')

    ########################EWC##########################
    parser.add_argument('--lambda', dest='lambda_', default=100, type=float,
                        help='EWC regularization coefficient')
    parser.add_argument('--alpha', dest='alpha', default=0.9, type=float,
                        help='EWC++ exponential moving average decay for Fisher calculation at each step')
    parser.add_argument('--fisher_update_after', dest='fisher_update_after', type=int, default=50,
                        help="Number of training iterations after which the Fisher will be updated.")

    ########################MIR#########################
    parser.add_argument('--subsample', dest='subsample', default=50,
                        type=int,
                        help='Number of subsample to perform MIR(default: %(default)s)')

    ########################GSS#########################
    parser.add_argument('--gss_mem_strength', dest='gss_mem_strength', default=10, type=int,
                        help='Number of batches randomly sampled from memory to estimate score')
    parser.add_argument('--gss_batch_size', dest='gss_batch_size', default=10, type=int,
                        help='Random sampling batch size to estimate score')

    ########################ASER########################
    parser.add_argument('--k', dest='k', default=5,
                        type=int,
                        help='Number of nearest neighbors (K) to perform ASER (default: %(default)s)')

    parser.add_argument('--aser_type', dest='aser_type', default="asvm", type=str, choices=['neg_sv', 'asv', 'asvm'],
                        help='Type of ASER: '
                             '"neg_sv" - Use negative SV only,'
                             ' "asv" - Use extremal values of Adversarial SV and Cooperative SV,'
                             ' "asvm" - Use mean values of Adversarial SV and Cooperative SV')

    parser.add_argument('--n_smp_cls', dest='n_smp_cls', default=2.0,
                        type=float,
                        help='Maximum number of samples per class for random sampling (default: %(default)s)')

    ########################CNDPM#########################
    parser.add_argument('--stm_capacity', dest='stm_capacity', default=1000, type=int, help='Short term memory size')
    parser.add_argument('--classifier_chill', dest='classifier_chill', default=0.01, type=float,
                        help='NDPM classifier_chill')
    parser.add_argument('--log_alpha', dest='log_alpha', default=-300, type=float, help='Prior log alpha')

    ########################GDumb#########################
    parser.add_argument('--minlr', dest='minlr', default=0.0005, type=float, help='Minimal learning rate')
    parser.add_argument('--clip', dest='clip', default=10., type=float,
                        help='value for gradient clipping')
    parser.add_argument('--mem_epoch', dest='mem_epoch', default=70, type=int, help='Epochs to train for memory')

    #######################Tricks#########################
    parser.add_argument('--labels_trick', dest='labels_trick', default=False, type=boolean_string,
                        help='Labels trick')
    parser.add_argument('--separated_softmax', dest='separated_softmax', default=False, type=boolean_string,
                        help='separated softmax')
    parser.add_argument('--kd_trick', dest='kd_trick', default=False, type=boolean_string,
                        help='Knowledge distillation with cross entropy trick')
    parser.add_argument('--kd_trick_star', dest='kd_trick_star', default=False, type=boolean_string,
                        help='Improved knowledge distillation trick')
    parser.add_argument('--review_trick', dest='review_trick', default=False, type=boolean_string,
                        help='Review trick')
    parser.add_argument('--ncm_trick', dest='ncm_trick', default=False, type=boolean_string,
                        help='Use nearest class mean classifier')
    parser.add_argument('--mem_iters', dest='mem_iters', default=1, type=int,
                        help='mem_iters')
    
    
    ####################Early Stopping######################
    parser.add_argument('--min_delta', dest='min_delta', default=0., type=float,
                        help='A minimum increase in the score to qualify as an improvement')
    parser.add_argument('--patience', dest='patience', default=0, type=int,
                        help='Number of events to wait if no improvement and then stop the training.')
    parser.add_argument('--cumulative_delta', dest='cumulative_delta', default=False, type=boolean_string,
                        help='If True, `min_delta` defines an increase since the last `patience` reset, '
                             'otherwise, it defines an increase after the last event.')

    ####################SupContrast######################
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--buffer_tracker', type=boolean_string, default=False,
                        help='Keep track of buffer with a dictionary')
    parser.add_argument('--warmup', type=int, default=4,
                        help='warmup of buffer before retrieve')
    parser.add_argument('--head', type=str, default='mlp',
                        help='projection head')
    parser.add_argument('--exp', type=str, default='PCR',
                        help='experiment name')
    parser.add_argument('--Triplet', dest='Triplet', default=False, type=boolean_string, help='Triplet_loss')
    parser.add_argument('--top5', dest='top5', default=False, type=boolean_string, help='Triplet loss use top5 as negative')
    parser.add_argument('--mem_bank_size', dest='mem_bank_size', default=1, type=int,
                        help='memory_bank_size')
    parser.add_argument('--num_subcentroids', dest='num_subcentroids', default=4, type=int,
                        help='num_subcentroids')
    parser.add_argument('--PSC', dest='PSC', default=False, type=boolean_string, help='PSC_loss')
    parser.add_argument('--onlyPSC', dest='onlyPSC', default=False, type=boolean_string, help='only use PSC_loss')
    parser.add_argument('--gamma', type=float, default=0.999,
                        help='weights for momentum update')
    
    ####################Lipschitz_Args######################
    parser.add_argument('--buffer_lip_lambda', type=float, required=False, default=0.5,
                        help='Lambda parameter for lipschitz minimization loss on buffer samples')
    
    # BUDGET LIP LOSS
    parser.add_argument('--budget_lip_lambda', type=float, required=False, default=0.5,
                        help='Lambda parameter for lipschitz budget distribution loss')

    # Extra
    parser.add_argument('--headless_init_act', type=str, choices=["relu","lrelu"], default="relu") #TODO:""
    parser.add_argument('--grad_iter_step', type=int, required=False, default=-2,
                            help='Step from which to enable gradient computation.') #TODO:""

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')
    parser.add_argument('--ignore_other_metrics', type=int, choices=[0, 1], default=0,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help="If set, run program with partial epochs and no wandb log.")
    
    
    ####################CLS_ER######################
    # Consistency Regularization Weight
    parser.add_argument('--reg_weight', type=float, default=0.1)

    # Stable Model parameters
    parser.add_argument('--stable_model_update_freq', type=float, default=0.70)
    parser.add_argument('--stable_model_alpha', type=float, default=0.999)

    # Plastic Model Parameters
    parser.add_argument('--plastic_model_update_freq', type=float, default=0.90)
    parser.add_argument('--plastic_model_alpha', type=float, default=0.999)

    parser.add_argument('--ucr_max', default=True, type=boolean_string,
                        help='Use maximal uncertainty or not')
    # parser.add_argument('--refresh', default=False, type=boolean_string,
    #                     help='Use refresh learning or not')

    parser.add_argument('--save_cp', dest='save_cp', default=False, type=boolean_string, help='save_checkpoint')
    # parser.add_argument('--cp_name', dest='cp_name', default='checkpoint.pth', type=str, help='checkpoint name')
    parser.add_argument('--cp_path', dest='cp_path', default='./checkpoint', type=str, help='checkpoint path')
    parser.add_argument('--hardness_analysis', dest='hardness_analysis', default=False, type=boolean_string, help='hardness_analysis')
    parser.add_argument('--analysis_path', dest='analysis_path', default='./analysis', type=str, help='analysis path')
    parser.add_argument('--buffer_analyze', default=False, type=boolean_string, help='buffer_analyze')
    parser.add_argument('--buffer_analyze_path', default='./buffer_analyze', type=str, help='buffer analyze path')
    parser.add_argument('--uniform_sampling', default=False, type=boolean_string, help='use uniform sampling')
    parser.add_argument('--return_logits', default=False, type=boolean_string, help='retrival_return_logits')
    # print(parser.data)
    # exit(0)
    initial_args, remaining_argv = parser.parse_known_args()
    # add_agent_parse(initial_args, parser)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    main(args)
