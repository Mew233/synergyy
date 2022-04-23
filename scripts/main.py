import argparse
from prepare_data import *
from select_features import *
from pipeline import *


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    parser.add_argument('--synergy_thres', type=int, default=0,
                        help='synergy threshold (default: loewe score)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='maximum number of epochs (default: 50)')
    parser.add_argument('--train_test_mode', type=str, default='train',
                        help='train or test')
    parser.add_argument('--model', type=str, default='LR',
                        help='import model (default: deepsynergy_preuer)')
                        #options are 'LR','XTBOOST','RF','ERT','deepsynergy_preuer','multitaskdnn_kim',
                        # 'matchmaker_brahim')

# --------------- Parse configuration  --------------- #

    parser.add_argument('--synergy_df', type=str, default='DrugComb')
    parser.add_argument('--drug_omics', nargs="+", default=['drug_target'],
                        required=False, help='drug target interaction/morgran fingerprints')    
    parser.add_argument('--cell_df', type=str, default='CCLE')
    parser.add_argument('--cell_omics', nargs="+", default=["exp"],
                        required=False, help='"exp","cn","mut"')
    parser.add_argument('--cell_filtered_by', type=str, default='variance',
                        help='top genes selected by variance or STRING graph')
    parser.add_argument('--get_cellfeature_concated', type=bool, default=True,
                        required=False, help='')
    parser.add_argument('--get_drugfeature_concated', type=bool, default=True,
                        required=False, help='if concat, numpy array')
    parser.add_argument('--get_drugs_summed', type=bool, default=True,
                        required=False, help='drug1+drug2 if True, else return dict')

    return parser.parse_args()


def main():
    args = arg_parse()

    write_config(args)
    X_cell, X_drug, Y = prepare_data(args)
    print("data loaded")
    model, test_loader = training(X_cell, X_drug, Y, args)
    print("training finished")
    val_results = evaluate(model, test_loader, args)
    print(' ROCAUC: {}, PRAUC: {}'.format(round(val_results['AUC'], 4),round(val_results['AUPR'], 4)))

    # save results
    # with open("results/%s.json"%(args.model), "w") as f:
    #     json.dump(val_results, f)


if __name__ == "__main__":
    main() 