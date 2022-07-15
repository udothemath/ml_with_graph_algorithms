"""
TODO: Refactor
- [X] Put this file into common as apply_bash.py
- [X] Allow customized input of:
    - [X] "'src.pd_lgd.model_apply'"
    - [X] 'ModelApply'
- [X] Allow customized input of
    - [X] apply_module (must)
    - [X] apply_class (must)
    - [X] input_table_name (must)
    - [X] target_table_name (must)
    - [X] test
    - [X] verbose
    - [X] batch_size
    - [X] parallel_count
    - [X] max_batch
- [X] add default variables
"""

import logging


def run(args):
    """
    讓apply可以被Bash Operator執行用的函式
    """
    import os
    import sys
    import gc
    from importlib import import_module
    current_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(current_path, '../../..')
    sys.path.insert(1, path)
    apply_cls = getattr(import_module(
        args.apply_module),
        args.apply_class)

    if args.max_batch != -1:
        apply_obj = apply_cls(
            group_id=args.group_id,
            verbose=args.verbose,
            max_batch=args.max_batch if args.max_batch != -1 else None)
    else:
        apply_obj = apply_cls(
            group_id=args.group_id,
            verbose=args.verbose)
    if args.test:
        offset = 0  # 起始行數
        limit = 10  # 挑選行數

        datamart_df = apply_obj.read_df_partially(offset, limit)
        logging.info(
            f'[apply_bash: run] Input DataFrame: {datamart_df.head(5)}')
        result_df = apply_obj.run_partially(offset, limit)
        logging.info(
            f'[apply_bash: run] Result DataFrame: {result_df.head(5)}')
        del datamart_df, result_df
        gc.collect()
        logging.info('[apply_bash: run] Clean up memory')
        apply_obj.close_conn()
        logging.info('[apply_bash: run] Connection Closed')

    apply_obj.run(
        parallel_count=args.parallel_count
    )


if __name__ in '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--apply_module',
        type=str,
        required=True,
        help='src.pd_lgd.model_apply'
    )
    parser.add_argument(
        '--apply_class',
        type=str,
        required=True,
        help='ModelApply'
    )

    parser.add_argument(
        '--group_id',
        type=int,
        required=False,
        help='assigning forking group id of the apply object',
        default=-1
    )

    parser.add_argument(
        '--test',
        type=bool,
        required=False,
        help='whether to test the apply function for the first 10 lines',
        default=True
    )

    parser.add_argument(
        '--verbose',
        type=bool,
        required=False,
        help='whether to execute message printing',
        default=True
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        required=False,
        help='number of rows applied per process during parallel processing',
        default=500
    )
    parser.add_argument(
        '--parallel_count',
        type=int,
        required=False,
        help='number of process',
        default=1
    )
    parser.add_argument(
        '--max_batch',
        type=int,
        required=False,
        help='row count connection test on airflow (None for all rows)',
        default=-1
    )

    args = parser.parse_args()
    options = vars(args)
    logging.info(options)
    run(args)
