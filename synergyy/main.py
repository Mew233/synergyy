import argparse
from prepare_data import *
from select_features import *
from pipeline import *
import os
os.environ["OMP_NUM_THREADS"] = "4"

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
    parser.add_argument('--model', type=str, default='TGSynergy',
                        help='import model (default: deepsynergy_preuer)')
                        #options are 'LR','XGBOOST','RF','ERT','deepsynergy_preuer','multitaskdnn_kim',
                        # 'matchmaker_brahim','deepdds_wang','TGSynergy)

# --------------- Parse configuration  --------------- #

    parser.add_argument('--synergy_df', type=str, default='DrugComb')
    parser.add_argument('--drug_omics', nargs="+", default=['smiles2graph_TGSynergy'],
                        required=False, help='drug_target/morgan_fingerprint/smiles2graph/smiles2graph_TGSynergy')    
    parser.add_argument('--cell_df', type=str, default='CCLE',
                        required=False,help='CCLE')
    parser.add_argument('--cell_omics', nargs="+", default=['GNN_cell'],
                        required=False, help='"exp","cn","mut","GNN_cell')
    parser.add_argument('--cell_filtered_by', type=str, default='STRING',
                        help='top genes selected by variance or STRING graph')
    parser.add_argument('--get_cellfeature_concated', type=bool, default=False,
                        required=False, help='')
    parser.add_argument('--get_drugfeature_concated', type=bool, default=False,
                        required=False, help='if concat, numpy array')
    parser.add_argument('--get_drugs_summed', type=bool, default=False,
                        required=False, help='drug1+drug2 if True, else return dict')

    return parser.parse_args()


def main():
    args = arg_parse()

    write_config(args)
    X_cell, X_drug, Y = prepare_data(args)
    print("data loaded")
    if args.model in ['LR','XGBOOST','RF','ERT']:
        model, scores, test_loader = training_baselines(X_cell, X_drug, Y, args)
        print("training finished")
        print(scores)
        val_results = evaluate(model, scores, test_loader, args)
        print("testing started")
        print(' ROCAUC: {}, PRAUC: {}'.format(round(val_results['AUC'], 4),round(val_results['AUPR'], 4)))

    # for deep learning models
    else:
        model, network_weights, test_loader = training(X_cell, X_drug, Y, args)
        print("training finished")
        val_results = evaluate(model, network_weights, test_loader, args)
        print("testing started")
        print(' ROCAUC: {}, PRAUC: {}'.format(round(val_results['AUC'], 4),round(val_results['AUPR'], 4)))
    
    # save results
    # with open("results/%s.json"%(args.model), "w") as f:
    #     json.dump(val_results, f)


if __name__ == "__main__":
    main() 