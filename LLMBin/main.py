# modified from https://github.com/ziyewang/COMEBin/blob/master/COMEBin/main.py
import argparse
import logging
import os
import pandas as pd
import datetime

from comebin_version import __version__ as ver
from train_CLmodel import train_CLmodel
from cluster import cluster
from arg_options import arguments


def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


# 日志时间改为北京时间
logging.Formatter.converter = beijing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """
    The main function of the COMEBin program.

    Functionality:
        - Initializes logging for the program.
        - Executes different subcommands based on user input.
        - Subcommands include: 'train', 'bin', 'nocontrast', 'generate_aug_data', and 'get_result'.
        - Subcommands perform various tasks such as data augmentation, training and clustering.
    """
    args = arguments()

    # logging
    logger = logging.getLogger('COMEBin\t'+ver)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y%m%d-%H:%M:%S')
    console_hdr = logging.StreamHandler()
    console_hdr.setFormatter(formatter)

    logger.addHandler(console_hdr)

    if args.subcmd == 'generate_aug_data':
        args.output_path = args.out_augdata_path

    os.makedirs(args.output_path, exist_ok=True)
    if args.subcmd == "generate_aug_data":
        handler = logging.FileHandler(args.output_path+f'/comebin_{args.model}_{args.llm_batch_size}bz_{args.contig_max_length}bp.log')
    else:
        handler = logging.FileHandler(args.output_path + f'/comebin.log')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    ## training
    if args.subcmd == 'train':
        logger.info('train')
        train_CLmodel(logger,args)

    ## clustering
    if args.subcmd == 'bin':
        logger.info('bin')
        from utils import gen_seed

        num_threads = args.num_threads
        _ = gen_seed(logger, args.contig_file, num_threads, args.contig_len, marker_name="bacar_marker", quarter="2quarter")

        cluster(logger, args)


    ## clustering NoContrast
    if args.subcmd == 'nocontrast':
        logger.info('NoContrast mode')
        from utils import get_kmer_coverage_aug0

        X_t, covMat, compositMat, namelist = get_kmer_coverage_aug0(args.data)

        X_t_df = pd.DataFrame(X_t, index=namelist)
        os.makedirs(args.output_path+'/combine_novars', exist_ok=True)
        outfile = args.output_path+'/combine_novars/combine_feature.tsv'
        X_t_df.to_csv(outfile, sep='\t', header=True)

        covMat_df = pd.DataFrame(covMat, index=namelist)
        os.makedirs(args.output_path+'/covMat', exist_ok=True)
        outfile = args.output_path+'/covMat/covMat_feature.tsv'
        covMat_df.to_csv(outfile, sep='\t', header=True)

        compositMat_df = pd.DataFrame(compositMat, index=namelist)
        os.makedirs(args.output_path+'/compositMat', exist_ok=True)
        outfile = args.output_path+'/compositMat/compositMat_feature.tsv'
        compositMat_df.to_csv(outfile, sep='\t', header=True)

        logger.info('NoContrast mode: generate features (aug0)')
        ori_outpath = args.output_path

        logger.info('NoContrast mode (combine) bin')
        args.output_path = ori_outpath +'/combine_novars'
        args.emb_file = args.output_path+'/combine_feature.tsv'
        cluster(logger,args)

        logger.info('NoContrast mode (coverage) bin')
        args.output_path = ori_outpath +'/covMat'
        args.emb_file = args.output_path+'/covMat_feature.tsv'
        cluster(logger,args)

        logger.info('NoContrast mode (kmer) bin')
        args.output_path = ori_outpath +'/compositMat'
        args.emb_file = args.output_path+'/compositMat_feature.tsv'
        cluster(logger,args)



    ##### generate_aug_data fastafile
    if args.subcmd == 'generate_aug_data':
        logger.info('generate_aug_data: fastafile')
        # 相关代码位于: /public/home/jialh/metaHiC/tools/BERTBin/BERTBin/data_aug
        # 总共产生6个视图，其中aug0是原始的contigs，其他的contigs通过打断原始的contigs产生。
        from data_aug.generate_augfasta_and_saveindex import run_gen_augfasta
        from data_aug.gen_cov import run_gen_cov
        from data_aug.gen_var import run_gen_cov_var

        logger.info(f"args in generate argument data: {args}")
        run_gen_augfasta(logger, args)
        # run_gen_cov: 实际上调用的是gen_cov.py第41行的gen_bedtools_out()函数，调用bedtools genomecov计算覆盖率。
        run_gen_cov(logger, args)
        run_gen_cov_var(logger, args)

    ###Generate the final results from the Leiden clustering results
    if args.subcmd == 'get_result':
        logger.info('get_result')
        from utils import gen_seed
        from get_final_result import run_get_final_result

        num_threads = args.num_threads
        seed_num = gen_seed(logger, args.contig_file, num_threads, args.contig_len, marker_name="bacar_marker", quarter="2quarter")

        run_get_final_result(logger, args, seed_num, num_threads, ignore_kmeans_res=True)


if __name__ == '__main__':
    main()

