"""Note!!!
This code is implemented on pytorch for evaluating deep learning models
"""
import os
import sys
sys.path.append(os.path.abspath('./utils'))
from engine import *
from model_bank import *
from multiscene import *
from multiscene_clean import *
import argparse

# **************** network params *************************
parser = argparse.ArgumentParser(description='multiscene_baseline')
parser.add_argument('--dataset', default='multiscene-clean', help='multiscene-clean or multiscene (default: MultiScene-Clean)')
parser.add_argument('--model_name', default='resnext101', help='vggnet16, vggnet19, inceptionv3, resnet50, resnet101, resnet152, squeezenet, mobilenetv2, shufflenetv2, densenet121, densenet169, resnetxt50, resnext101, nasnet, lr-vggnet16, lr-resnet50 (default: resnext101)')
parser.add_argument('--pretrain', action='store_true', help='use pretrain models or not (default: False)')
parser.add_argument('--nb_moda', default=32, type=int, help='number of modality per label for label relation network')
parser.add_argument('--nb_units', default=32, type=int, help='parameters for label relation network')
parser.add_argument('--nb_classes', default=36, type=int, help='MultiScene and MultiScene-Clean have 36 scene classes (default: 36)')
parser.add_argument('--weight_path', default='weights/weights.h5')
parser.add_argument('--image_size', default=224, type=int, help='the size of input image (default: 224)')
parser.add_argument('--batch_size', default=16, type=int, help='the size of mini-batch (default: 16)')
parser.add_argument('--lr', default=2e-2, type=float, help='learning rate (default: 0.02)')
parser.add_argument('--lrp', default=1, type=float, help='learning rate decay for non-backbone layers(default: 1)')
parser.add_argument('--epochs', default=200, type=int, help='training epochs (default: 100)')
parser.add_argument('--evaluate', action='store_true', help='evaluate model (default: False)')
parser.add_argument('--workers', default=4, type=int, help='number of parallel workers (default: 4)')

# ********************** main ******************************
def main():

    args = parser.parse_args()

    # define dataset
    if args.dataset=='multiscene-clean':
        train_dataset = MultiSceneClean('Tra')
        #val_dataset = MultiSceneClean('Val') # split from Tra.csv
        test_dataset = MultiSceneClean('Test')
    elif args.dataset=='multiscene':
        train_dataset = MultiScene('Tra')
        #val_dataset = MultiScene('Val') # split from Tra.csv
        test_dataset = MultiScene('Test')

    # load model
    model = build_model(args)

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    state = {'save_model_path': os.path.join('weights', args.dataset, args.model_name),
             'batch_size': args.batch_size, 
             'image_size': args.image_size, 
             'max_epochs': args.epochs,
             'evaluate': args.evaluate, 
             'resume': args.weight_path,
             'num_classes': args.nb_classes,
             'workers': args.workers,
             'epoch_step': 100,
             'lr': args.lr,
             'use_gpu': torch.cuda.is_available()}
    
    state['print_freq'] = 0
    state['print_epoch'] = 30
    engine = MultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, test_dataset, optimizer)



if __name__ == '__main__':
    main()
