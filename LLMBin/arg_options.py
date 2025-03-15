import argparse
from llmbin_version import __version__ as ver


def arguments():
    """
    LLMBin: A contig binning method based on Language model and Contrastive Multi-view representation learning.

    :return: Parsed command-line arguments
    """
    doc = f"""COMEBin: a contig binning method based on COntrastive Multi-viEw representation learning.
    Version:{ver}"""
    parser = argparse.ArgumentParser(
        prog="llmbin",
        description=doc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage="%(prog)s outdir contig_file path_to_bamfiles [options]")
    parser.version = ver

    parser.add_argument('-v','--version',
                        action='version',
                        help='COMEBin version')

    subparsers = parser.add_subparsers(title='COMEBin subcommands',
                                       dest='subcmd',
                                       metavar='')

    #############################################################################################
    ############################################ CLtraining #####################################
    ### Command-line arguments and options for training the network.

    CLtraining_subparsers = subparsers.add_parser('train',
                                                  help='Train the model based on the augmentation data')

    CLtraining_subparsers.add_argument('--data', metavar='DIR', default='/home/wzy/data/STEC_data/for_new_method/data_augmentation_clean/',
                        help='path to dataset')
    CLtraining_subparsers.add_argument('--kmer_model_path', metavar='DIR', default='empty',
                        help='kmer_model_path')
    CLtraining_subparsers.add_argument('--output_path', metavar='DIR', default='output',
                                       help='Output path.')
    CLtraining_subparsers.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                        help='number of data loading workers (default: 5)')
    CLtraining_subparsers.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    CLtraining_subparsers.add_argument('--covmodelepochs', default=50, type=int, metavar='N',
                        help='number of total epochs to pretrain coverage model')

    CLtraining_subparsers.add_argument('--model_max_length', default=5000, type=int, metavar='N',
                                              help='mininum contig length for augmentation')
    CLtraining_subparsers.add_argument('--model', default="dnabert2", type=str, metavar='N',
                                              help='model for LLM embedding (hyenadna, dnabert2, nt, dnabert_s)')

    CLtraining_subparsers.add_argument('-b', '--batch_size', default=1024, type=int,
                        metavar='N',
                        help='mini-batch size (default: 1024), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    CLtraining_subparsers.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    CLtraining_subparsers.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    CLtraining_subparsers.add_argument('--out_dim', default=128, type=int,
                        help='feature dimension (default: 128)')

    CLtraining_subparsers.add_argument('--dropout_value', default=0.2, type=float,
                        help='dropout_value (default: 0.2)')
    CLtraining_subparsers.add_argument('--emb_szs', default=2048, type=int,
                        help='embedding size for hidden layer (default: 2048)')
    CLtraining_subparsers.add_argument('--n_layer', default=3, type=int,
                        help='n layers (default: 3)')
    CLtraining_subparsers.add_argument('--add_model_for_coverage', action="store_true",
                        help='add_model_for_coverage.')

    CLtraining_subparsers.add_argument('--nokmer', action="store_true",
                        help='only use coverage information.')

    CLtraining_subparsers.add_argument('--pretrain_coveragemodel', action="store_true",
                        help='train ')

    CLtraining_subparsers.add_argument('--emb_szs_forcov', default=2048, type=int,
                        help='embedding size for hidden layer (default: 2048)')
    CLtraining_subparsers.add_argument('--out_dim_forcov', default=128, type=int,
                        help='embedding size for hidden layer (default: 128)')
    CLtraining_subparsers.add_argument('--n_layer_forcov', default=3, type=int,
                        help='n_layer_forcov (default: 3)')

    CLtraining_subparsers.add_argument('--emb_szs_forllm', default=2048, type=int,
                                       help='embedding size for hidden layer (default: 2048)')
    CLtraining_subparsers.add_argument('--out_dim_forllm', default=128, type=int,
                                       help='embedding size for hidden layer (default: 128)')
    CLtraining_subparsers.add_argument('--n_layer_forllm', default=3, type=int,
                                       help='n_layer_forllm (default: 3)')


    CLtraining_subparsers.add_argument('--earlystop', action="store_true",
                                       help='earlystop.')

    CLtraining_subparsers.add_argument('--pretrain_kmer_model_path', metavar='DIR', default='no',
                        help='pretrain_kmer_model_path')

    CLtraining_subparsers.add_argument('--not_load_kmermetric_state', action="store_true",
                        help='not_load_kmermetric_state')

    CLtraining_subparsers.add_argument('--finetunepretrainmodel', action="store_true",
                        help='finetunepretrainmodel')
    CLtraining_subparsers.add_argument('--finetunelr_ratio', default=0.1, type=float, help='finetune learning rate')

    CLtraining_subparsers.add_argument('--addvars', action="store_true",
                                       help='addvars')
    CLtraining_subparsers.add_argument('--vars_sqrt', action="store_true",
                                       help='vars_sqrt')

    CLtraining_subparsers.add_argument('--log-every-n-steps', default=20, type=int,
                        help='Log every n steps')

    CLtraining_subparsers.add_argument('--temperature', default=0.1, type=float,
                        help='softmax temperature (default: 0.1)')
    CLtraining_subparsers.add_argument('--covmodel_temperature', default=0.1, type=float,
                        help='softmax temperature (default: 0.1)')
    CLtraining_subparsers.add_argument('--n_views', default=6, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    CLtraining_subparsers.add_argument('--contig_len', default = 1000, type=int, metavar='N',
                        help='mininum contig length for training')
    CLtraining_subparsers.add_argument('--notuse_scheduler', action='store_true',
                        help='notuse_scheduler')
    CLtraining_subparsers.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    CLtraining_subparsers.add_argument('--addcovloss', action='store_true',
                        help='addcovloss')
    CLtraining_subparsers.add_argument('--lambdaloss2', default=0.1, type=float,
                        help='lambdaloss2 (default: 1)')

    CLtraining_subparsers.add_argument('--addkmerloss', action='store_true',
                                       help='addkmerloss')
    CLtraining_subparsers.add_argument('--lambdakmerloss2', default=0.1, type=float,
                                       help='lambdakmerloss2 (default: 1)')
    CLtraining_subparsers.add_argument('--kmermodel_temperature', default=0.1, type=float,
                                       help='kmermodel_temperature softmax temperature (default: 0.1)')

    CLtraining_subparsers.add_argument('--addllmloss', action='store_true',
                                       help='addllmloss')
    CLtraining_subparsers.add_argument('--lambdallmloss2', default=0.5, type=float,
                                       help='lambdallmloss2 (default: 1)')
    CLtraining_subparsers.add_argument('--llmmodel_temperature', default=0.1, type=float,
                                       help='llmmodel_temperature softmax temperature (default: 0.1)')

    CLtraining_subparsers.add_argument('--cov_meannormalize', action='store_true',
                                       help='cov_meannormalize')
    CLtraining_subparsers.add_argument('--cov_minmaxnormalize', action='store_true',
                                       help='cov_minmaxnormalize')
    CLtraining_subparsers.add_argument('--cov_standardization', action='store_true',
                                       help='cov_standardization')
    CLtraining_subparsers.add_argument('--notuseaug0', action='store_true',
                                       help='notuseaug0')
    CLtraining_subparsers.add_argument('--lossbalance', action='store_true',
                                       help='lossbalance ((whole_loss[epoch0]/covloss[epoch0])')

    CLtraining_subparsers.add_argument('--kmer_l2_normalize', action='store_true',
                                       help='kmer_l2_normalize (used for nokmerMetric)')

    CLtraining_subparsers.add_argument('--kmerMetric_notl2normalize', action='store_true',
                                       help='kmerMetric_notl2normalize (used for kmerMetric)')
    CLtraining_subparsers.add_argument('--covmodel_notl2normalize', action='store_true',
                                       help='covmodel_notl2normalize (used for covmodel)')
    CLtraining_subparsers.add_argument('--llmmodel_notl2normalize', action='store_true',
                                       help='llmmodel_notl2normalize (used for llmmodel)')
    CLtraining_subparsers.add_argument('--num_threads', default=10, type=int,
                                              help='num_threads for training in CPU mode.')



    #############################################################################################
    ############################################ cluster NoContrast #####################################
    ### Command-line arguments and options for running the COMEBin using the original features.

    NoContrast_subparsers = subparsers.add_parser('nocontrast',
                                                   help='Cluster the contigs using original features.')

    NoContrast_subparsers.add_argument('--contig_file', type=str, help=("The contigs file."))
    NoContrast_subparsers.add_argument('--seed_file', type=str, help=("The marker seed file."))
    NoContrast_subparsers.add_argument('--data', metavar='DIR', default='/home/wzy/data/STEC_data/for_new_method/data_augmentation_clean/',
                                       help='path to dataset')
    NoContrast_subparsers.add_argument('--output_path', type=str, default='temp_output', help=("The output path"))

    NoContrast_subparsers.add_argument('--cluster_num', default=0, type=int,
                                       help='Add cluster number to run partial seed method (default: 0)')

    NoContrast_subparsers.add_argument('--not_run_infomap', action='store_true',
                                       help='Do not run infomap.')

    NoContrast_subparsers.add_argument('--not_l2normaize', action='store_true',
                                       help='Do not run l2normaize for embeddings.')

    NoContrast_subparsers.add_argument('--contig_len', default = 1001, type=int, metavar='N',
                                       help='mininum contig length for clustering')



    #############################################################################################
    ############################################ generate aug data #####################################
    ### Command-line arguments and options for data augmentation.

    generate_aug_data_subparsers = subparsers.add_parser('generate_aug_data',
                                                      help='Generate the augmentation data and features from the fasta file and bam files.')
    generate_aug_data_subparsers.add_argument('--contig_file', type=str, help=("The original contigs file."))
    generate_aug_data_subparsers.add_argument('--out_augdata_path', type=str, help=("The output path to save the augmentation data"))
    generate_aug_data_subparsers.add_argument('--n_views', default=6, type=int,
                                           help='n_views for generating augmentation data.')
    generate_aug_data_subparsers.add_argument('--bam_file_path', type=str, help=("The path to access the bam files."))
    generate_aug_data_subparsers.add_argument('--contig_len', default = 1000, type=int, metavar='N',
                                       help='mininum contig length for augmentation')
    generate_aug_data_subparsers.add_argument('--llm_batch_size', default=16, type=int, metavar='N',
                                              help='batch_size for LLMs model')
    generate_aug_data_subparsers.add_argument('--model_max_length', default=5000, type=int, metavar='N',
                                              help='max length for LLMs model')
    generate_aug_data_subparsers.add_argument('--contig_max_length', default=10000, type=int, metavar='N',
                                              help='max contig length for augmentation')
    generate_aug_data_subparsers.add_argument('--model', default="dnabert2", type=str, metavar='N',
                                              help='model for LLM embedding (hyenadna, dnabert2, nt, dnabert_s)')
    generate_aug_data_subparsers.add_argument('--test_model_dir', default="./", type=str, metavar='N',
                                              help='model for LLM embedding (hyenadna, dnabert2, nt, dnabert_s)')
    generate_aug_data_subparsers.add_argument('--num_threads', default=10, type=int,
                                              help='num_threads for generating augmentation data.')

    #############################################################################################
    ############################################ cluster #####################################
    ### Command-line arguments and options for running the Leiden-based clustering.

    clustering_subparsers = subparsers.add_parser('bin',
                                                  help='Cluster the contigs.')

    clustering_subparsers.add_argument('--contig_file', type=str, help=("The contigs file."))
    clustering_subparsers.add_argument('--seed_file', type=str, help=("The marker seed file."))
    clustering_subparsers.add_argument('--emb_file', type=str, help=("The embedding feature file."))
    clustering_subparsers.add_argument('--output_path', type=str, help=("The output path"))

    clustering_subparsers.add_argument('--cluster_num', default=0, type=int,
                                       help='Add cluster number to run partial seed method (default: 0)')

    clustering_subparsers.add_argument('--not_run_infomap', action='store_true',
                                       help='Do not run infomap.')
    clustering_subparsers.add_argument('--not_l2normaize', action='store_true',
                                       help='Do not run l2normaize for embeddings.')


    clustering_subparsers.add_argument('--contig_len', default = 1001, type=int, metavar='N',
                                       help='mininum contig length for clustering')
    clustering_subparsers.add_argument('--num_threads', default=10, type=int,
                                              help='num_threads for binning.')


    #############################################################################################
    ############################################ get final results #####################################
    ### Command-line arguments and options for generating the final binning result from the Leiden clustering results.

    get_result_subparsers = subparsers.add_parser('get_result',
                                                  help='Generate the final results from the Leiden clustering results.')

    get_result_subparsers.add_argument('--contig_file', type=str, help=("The contigs file."))
    get_result_subparsers.add_argument('--seed_file', type=str, help=("The marker seed file."))
    get_result_subparsers.add_argument('--emb_file', type=str, help=("The embedding feature file."))
    get_result_subparsers.add_argument('--output_path', type=str, help=("The output path"))
    get_result_subparsers.add_argument('--binning_res_path', type=str, help=("The path to get Leiden clustering results"))
    get_result_subparsers.add_argument('--contig_len', default = 1001, type=int, metavar='N',
                                       help='mininum contig length for clustering')
    get_result_subparsers.add_argument('--num_threads', default=10, type=int,
                                       help='num_threads for getting final result.')

    get_result_subparsers.add_argument('--bac_mg_table', type=str, help=("bac_mg_table (bacteria marker gene information)"))
    get_result_subparsers.add_argument('--ar_mg_table', type=str, help=("ar_mg_table (archea marker gene information)"))

    args = parser.parse_args()
    return args