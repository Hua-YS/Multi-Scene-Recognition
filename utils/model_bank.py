from cnn import *
from saff import *
from facnn import *
from kfb import *
from relationnet import *
from backbone import *
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import torchvision.models as models

def build_model(args):
    model_name = args.model_name
    nb_classes = args.nb_classes
    pretrained = args.pretrain
    if model_name == 'vggnet16':
        return VGGNetBaseline(models.vgg16(pretrained=pretrained), nb_classes)
    elif model_name == 'vggnet19':
        return VGGNetBaseline(models.vgg19(pretrained=pretrained), nb_classes)
    elif model_name == 'inceptionv3':
        return InceptionBaseline(models.inception_v3(pretrained=pretrained), nb_classes)
    elif model_name == 'resnet50':
        return ResNetBaseline(models.resnet50(pretrained=pretrained), nb_classes)
    elif model_name == 'resnet101':
        return ResNetBaseline(models.resnet101(pretrained=pretrained), nb_classes)
    elif model_name == 'resnet152':
        return ResNetBaseline(models.resnet152(pretrained=pretrained), nb_classes)
    elif model_name == 'squeezenet':        
        return SqueezeNetBaseline(models.squeezenet1_0(pretrained=pretrained), nb_classes)
    elif model_name == 'densenet121':        
        return DenseNetBaseline(models.densenet121(pretrained=pretrained), nb_classes)
    elif model_name == 'densenet169':
        return DenseNetBaseline(models.densenet169(pretrained=pretrained), nb_classes)
    elif model_name == 'shufflenetv2':
        return ShuffleNetBaseline(models.shufflenet_v2_x1_0(pretrained=pretrained), nb_classes)
    elif model_name == 'mobilenetv2':
        return MobileNetBaseline(models.mobilenet_v2(pretrained=pretrained), nb_classes)
    elif model_name == 'resnext50':
        return ResNeXtBaseline(models.resnext50_32x4d(pretrained=pretrained), nb_classes)
    elif model_name == 'resnext101':
        return ResNeXtBaseline(models.resnext101_32x8d(pretrained=pretrained), nb_classes)
    elif model_name == 'mnasnet':
        return MNASNetBaseline(models.mnasnet1_0(pretrained=pretrained), nb_classes)
    elif model_name == 'lr-vggnet16':
        return RelationNet('vggnet16', nb_classes, num_moda=args.nb_moda, num_units=args.nb_units)
    elif model_name == 'lr-resnet50':
        return RelationNet('resnet50', nb_classes, num_moda=args.nb_moda, num_units=args.nb_units)
    elif model_name == 'svm':
        return MultiOutputClassifier(SVC(random_state=0, tol=1e-5, max_iter=100000, verbose=1), -1)
    elif model_name == 'xgboost':
        return MultiOutputClassifier(XGBClassifier(booster='gbtree', n_jobs=100, n_estimators=200, verbosity=1, use_label_encoder=False, gpu_id=0), -1)
    elif model_name == 'rf':
        return MultiOutputClassifier(RandomForestClassifier(random_state=0, n_estimators=200, verbose=1), -1)
    elif model_name == 'saff':
        return SAFF(models.vgg16(pretrained=pretrained), nb_classes, 256)
    elif model_name == 'facnn':
        return FACNN(models.vgg16(pretrained=pretrained), nb_classes)
    elif model_name == 'kfb':
        return KFB(models.vgg16(pretrained=pretrained), nb_classes)
    else:
        print('The selected model is not pre-defined! Now got to default model (ResNeXt101)!')
        return ResNeXtBaseline(models.resnext101_32x8d(pretrained=pretrained), nb_classes)


