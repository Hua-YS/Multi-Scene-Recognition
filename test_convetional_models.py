"""Note!!!
This code is implemented using scikit for evaluating SVM, RF, XGBOOST
"""
import os
import sys
sys.path.append(os.path.abspath('./utils'))
from metrics import *
from model_bank import *
from multiscene import *
from multiscene_clean import *
import argparse

# **************** network params *************************
parser = argparse.ArgumentParser(description='multiscene_sklearn')
parser.add_argument('--dataset', default='multiscene-clean', help='multiscene-clean or multiscene (default: MultiScene-Clean)')
parser.add_argument('--model_name', default='svm', help='svm, xgboost, rf (default: svm)')
parser.add_argument('--pretrain', action='store_true', help='use pretrain models or not (default: False)')
parser.add_argument('--nb_classes', default=36, type=int, help='MultiScene and MultiScene-Clean have 36 scene classes (default: 36)')
parser.add_argument('--weight_path', default='weights/weights.h5')

# ********************** main ******************************
def main():

    args = parser.parse_args()

    # define dataset
    if args.dataset=='multiscene-clean':
        x_tra, y_tra = MultiSceneClean_sklearn('Tra')
        x_test, y_test = MultiSceneClean_sklearn('Test')
    elif args.dataset=='multiscene':
        x_tra, y_tra = MultiScene_sklearn('Tra')
        x_test, y_test = MultiScene_sklearn('Test')

    # load model, loss is defined inside
    model = build_model(args)
    model.fit(x_tra, y_tra)
    
    y_pred = model.predict(x_test)*2-1
    meter = AveragePrecisionMeter(False)
    meter.add(y_pred, y_test)
    print('per-class AP:', meter.value()*100)
    print('mAP:', meter.value().mean()*100)
    OP, OR, OF1, CP, CR, CF1, EP, ER, EF1 = meter.overall()
    print('CP | CR | CF1 | EP | ER | EF1 | OP | OR | OF1\n'
          '---------------------------------------------\n'
          '{CP:.1f}\t'
          '{CR:.1f}\t'
          '{CF1:.1f}\t'
          '{EP:.1f}\t'
          '{ER:.1f}\t'
          '{EF1:.1f}\t'
          '{OP:.1f}\t'
          '{OR:.1f}\t'
          '{OF1:.1f}'.format(CP=CP*100, CR=CR*100, CF1=CF1*100, EP=EP*100, ER=ER*100, EF1=EF1*100, OP=OP*100, OR=OR*100, OF1=OF1*100))        
    print('==========================================================\n')


if __name__ == '__main__':
    main()
