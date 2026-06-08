from __future__ import print_function

from networks.gol import GOL
from networks.gol_learnable_sigma import GOL_learnableSigma
from networks.gol_adaptive_sigma import GOL_adaptiveSigma

def prepare_model(opt):
    model = eval(opt.model)(opt)
    return model




def prepare_model_sigma(opt):
    model = eval(opt.model_sigma)(opt)
    return model

