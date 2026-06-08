import torchvision.transforms as transforms
from PIL import Image

class ConfigBasic:
    def __init__(self,):
        self.dataset = None
        self.setting = None
        self.logscale = False
        self.set_optimizer_parameters()
        self.set_training_opts()
        self.set_network()

    def set_dataset(self):
        self.is_filelist = True
        if self.dataset == 'morph':
            if self.logscale:
                self.tau = 0.1
            else:
                self.tau = 2

            self.img_root = '/hdd1/cwlee/datasets/OrderLearning/img/MORPH'
            if self.setting == 'A':
                self.is_filelist = True
                if self.noised:
                    if self.noise_type == 'Gaussian':
                        if self.noise_percentage ==50:#self.noise_percentage == 30 or  
                            self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_Setting{self.setting}_noised_alpha1_std{self.noise_percentage}%_v2/Setting{self.setting}_fold{self.fold}_train.txt'
                            self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_Setting{self.setting}_noised_alpha1_std{self.noise_percentage}%_v2/Setting{self.setting}_fold{self.fold}_test.txt'
                        # elif self.noise_percentage ==30:#self.noise_percentage == 30 or  
                        #     self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_Setting{self.setting}_noised_alpha1_std{self.noise_percentage}%_NEW/Setting{self.setting}_fold{self.fold}_train.txt'
                        #     self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_Setting{self.setting}_noised_alpha1_std{self.noise_percentage}%_NEW/Setting{self.setting}_fold{self.fold}_test.txt'
                        else:
                            # self.train_file = '/ssd1/cwlee/ICLR2025_F/morph/settingA_tau3/noise40%/WARMUP_V2_CentroidFixed_noisedTrue40%_SOL_DiscrTrueL1_pairwise_lr0.0001_refinement_noiseRate0.9_correct_constant_rate_mean/PREFIX_0.25_tau3_F0_GOL_vgg16v2norm_2024-09-29 20:33:27/updated.csv'
                            # self.train_file = '/home/cwlee/ICLR2026/SOL/morph/settingA_tau3/noise20%/WARMUP_V2_CentroidFixed_noisedTrue20%_SOL_DiscrTrueL1_pairwise_lr0.0001_refinement_noiseRate0.9_correct_constant_rate_mean/PREFIX_0.25_tau3_F0_GOL_vgg16v2norm_2024-09-29 19:10:22/updated.csv'
                            self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_Setting{self.setting}_noised_alpha1_std{self.noise_percentage}%/Setting{self.setting}_fold{self.fold}_train.txt'
                            self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_Setting{self.setting}_noised_alpha1_std{self.noise_percentage}%/Setting{self.setting}_fold{self.fold}_test.txt'
                      
                    elif self.noise_type == 'input_dependent':
                        self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_Setting{self.setting}_IDN_{self.idn_variant}_noised_alpha1_std{self.noise_percentage}%_v3/Setting{self.setting}_fold{self.fold}_train.txt'
                        self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_Setting{self.setting}_IDN_{self.idn_variant}_noised_alpha1_std{self.noise_percentage}%_v3/Setting{self.setting}_fold{self.fold}_test.txt'
                    
                    else:
                        self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_Setting{self.setting}_{self.noise_type}_noised_alpha1_std{self.noise_percentage}%_v2/Setting{self.setting}_fold{self.fold}_train.txt'
                        self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_Setting{self.setting}_{self.noise_type}_noised_alpha1_std{self.noise_percentage}%_v2/Setting{self.setting}_fold{self.fold}_test.txt'
                    self.noised_lb_idx = 6
                else:
                    self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_Setting{self.setting}/Setting{self.setting}_fold{self.fold}_train.txt'
                    self.test_file =  f'/hdd1/cwlee/datasets/OrderLearning/index/MORPH_Setting{self.setting}/Setting{self.setting}_fold{self.fold}_test.txt'
                self.delimeter = ","
                self.img_idx = 4
                self.lb_idx = 3

       
        elif self.dataset == 'adience':
            self.is_filelist = False
            if self.noised:
                self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience_noised_alpha1_std{self.noise_percentage}_V3/adience_F{self.fold}_train_algn_[0_7].pickle'
                self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience_noised_alpha1_std{self.noise_percentage}_V3/adience_F{self.fold}_test_algn_[0_7].pickle'
            else:
                self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.fold}_train_algn_[0_7].pickle'
                self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/Adience/adience_F{self.fold}_test_algn_[0_7].pickle'

            # self.train_file = f'/hdd/2021/Research/99_dataset/Adience/adience_F{self.fold}_train_algn_[0_7].pickle'
            # self.test_file = f'/hdd/2021/Research/99_dataset/Adience/adience_F{self.fold}_test_algn_[0_7].pickle'
            self.tau = 1

        elif self.dataset =='clap':
            self.is_filelist = True
            self.img_root = '/hdd1/cwlee/datasets/OrderLearning/img/CLAP/2015'
            if self.noised:
                self.delimeter = None
                self.img_idx = 1
                self.lb_idx = 2
                self.data_type_idx = 4
                self.std_idx = 3
                if self.fold == 'eval_on_test':
                    if self.noise_type != 'Gaussian':
                        self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/clap_split_{self.noise_type}_noised_alpha1_std{self.noise_percentage}%_v2/CLAP_trainval.txt'
                        self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/clap_split_{self.noise_type}_noised_alpha1_std{self.noise_percentage}%_v2/CLAP_test.txt'
                    else:
                        if self.noise_percentage == 20 or self.noise_percentage == 10 or self.noise_percentage == 50:
                            self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/clap_split_noised_alpha1_std{self.noise_percentage}%/CLAP_trainval.txt'
                            self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/clap_split_noised_alpha1_std{self.noise_percentage}%/CLAP_test.txt'
                        else:
                            self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/clap_split_noised_alpha1_std{self.noise_percentage}%_v2/CLAP_trainval.txt'
                            self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/clap_split_noised_alpha1_std{self.noise_percentage}%_v2/CLAP_test.txt'
                elif self.fold == 'eval_on_val':
                    self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/index/clap_split_noised_alpha1_std{self.noise_percentage}/CLAP_train.txt'
                    self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/index/clap_split_noised_alpha1_std{self.noise_percentage}/CLAP_val.txt'
                else:
                    raise ValueError(f'check fold: it should be [eval_on_test] or [eval_on_val], but {self.fold} is given.')
            else:
                self.delimeter = " "
                self.img_idx = 0
                self.lb_idx = 1
                self.data_type_idx = 3
                self.std_idx = 2
                if self.fold == 'eval_on_test':
                    self.train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_trainval.txt'
                    self.test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_test.txt'
                elif self.fold == 'eval_on_val':
                    self.train_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_train.txt'
                    self.test_file = '/hdd1/cwlee/datasets/OrderLearning/index/clap_split/CLAP_val.txt'
                else:
                    raise ValueError(
                        f'check fold: it should be [eval_on_test] or [eval_on_val], but {self.fold} is given.')
        ###################################################################################################################
        elif self.dataset == 'aadb':
            self.img_root = '/hdd1/cwlee/datasets/OrderLearning/AADB/datasetImages_originalSize'
            self.is_filelist = True
            if self.noise_type == 'Gaussian':
                if self.noised:#v2 for ?
                    self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/AADB/ImageAesthetics_ECCV2016/imgListFiles_label_noised_alpha1_std{self.noise_percentage}%/imgListTrainRegression_score.txt'
                    self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/AADB/ImageAesthetics_ECCV2016/imgListFiles_label_noised_alpha1_std{self.noise_percentage}%/imgListTestRegression_score.txt'
                else:
                    self.train_file = '/hdd1/cwlee/datasets/OrderLearning/AADB/ImageAesthetics_ECCV2016/imgListFiles_label/imgListTrainRegression_score.txt'
                    self.test_file = '/hdd1/cwlee/datasets/OrderLearning/AADB/ImageAesthetics_ECCV2016/imgListFiles_label/imgListTestRegression_score.txt'
            else:
                if self.noised:#v2 for ?
                    self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/AADB/ImageAesthetics_ECCV2016/imgListFiles_label_{self.noise_type}_noised_alpha1_std{self.noise_percentage}%_v2/imgListTrainRegression_score.txt'
                    self.test_file = f'/hdd1/cwlee/datasets/OrderLearning/AADB/ImageAesthetics_ECCV2016/imgListFiles_label_{self.noise_type}_noised_alpha1_std{self.noise_percentage}%_v2/imgListTestRegression_score.txt'
                else:
                    self.train_file = '/hdd1/cwlee/datasets/OrderLearning/AADB/ImageAesthetics_ECCV2016/imgListFiles_label/imgListTrainRegression_score.txt'
                    self.test_file = '/hdd1/cwlee/datasets/OrderLearning/AADB/ImageAesthetics_ECCV2016/imgListFiles_label/imgListTestRegression_score.txt'

            # self.train_file = '/hdd/2020/Research/datasets/AADB/ImageAesthetics_ECCV2016/imgListFiles_label/imgListTrainRegression_score.txt'
            # self.test_file = '/hdd/2020/Research/datasets/AADB/ImageAesthetics_ECCV2016/imgListFiles_label/imgListTestRegression_score.txt'
            # self.train_file = '/hdd/2023/v3_results/aadb/aadb_toy1000.txt'
            # self.test_file = '/hdd/2023/v3_results/aadb/aadb_toy1000.txt'
            self.tau = 0.1
            self.delimeter = None
            self.img_idx = 0
            self.lb_idx = 1
        #############################################################################################################################################################
        elif self.dataset == 'rsna':
            self.ds_root = '/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets'
            self.delimeter = None
            self.is_filelist=True
            if self.noised:
                if self.noise_type == 'Gaussian':
                    self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets/Bone Age Training Set/train_noised_alpha1_std{self.noise_percentage}%.csv'
                    self.test_file = '/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets/Bone Age Test Set/Bone age ground truth.xlsx'
                    # self.test_file = '/hdd1/cwlee/datasets/OrderLearning/Bone Age DatasetsBone Age Validation Set/Validation Dataset.csv'
                else:
                    self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets/Bone Age Training Set/train_{self.noise_type}_noised_alpha1_std{self.noise_percentage}%.csv'
                    self.test_file = '/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets/Bone Age Test Set/Bone age ground truth.xlsx'
                    # self.test_file = '/hdd1/cwlee/datasets/OrderLearning/Bone Age DatasetsBone Age Validation Set/Validation Dataset.csv'
            else:
                self.train_file ='/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets/Bone Age Training Set/train.csv'
                self.test_file = '/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets/Bone Age Test Set/Bone age ground truth.xlsx'
                # self.test_file = '/hdd1/cwlee/datasets/OrderLearning/Bone Age DatasetsBone Age Validation Set/Validation Dataset.csv'
        elif self.dataset == 'rsna_toy':
            self.ds_root = '/hdd1/cwlee/datasets/OrderLearning/Bone Age Datasets'
            self.is_filelist = True
            if self.noised:
                self.train_file = f'/hdd1/cwlee/datasets/OrderLearning/Bone Age DatasetsBone Age Training Set/train_noised_alpha1_std{self.noise_percentage}_TOY.csv'
                self.test_file = '/hdd1/cwlee/datasets/OrderLearning/Bone Age DatasetsBone Age Test Set/Bone age ground truth.xlsx'
                # self.test_file = '/hdd1/cwlee/datasets/OrderLearning/Bone Age DatasetsBone Age Validation Set/Validation Dataset.csv'
            else:
                self.train_file = '/hdd1/cwlee/datasets/OrderLearning/Bone Age DatasetsBone Age Training Set/train.csv'
                self.test_file = '/hdd1/cwlee/datasets/OrderLearning/Bone Age DatasetsBone Age Test Set/Bone age ground truth.xlsx'
                # self.test_file = '/hdd1/cwlee/datasets/OrderLearning/Bone Age DatasetsBone Age Validation Set/Validation Dataset.csv'
        ##############################################################################################################################################################3
        else:
            raise ValueError(f'{self.dataset} is out of range!')

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform_tr = transforms.Compose([
                                                lambda x: Image.fromarray(x),
                                                transforms.RandomCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                self.normalize
                                                ])

        self.transform_te = transforms.Compose([
                                                lambda x: Image.fromarray(x),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                self.normalize
                                                ])

        self.transform_vis = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.CenterCrop(224),
            #transforms.ToTensor(),
        ])


    def set_optimizer_parameters(self):
        # *** Optimizer
        self.adam = True
        self.learning_rate = 0.0001
        self.lr_decay_epochs = [30, 50, 100]
        self.lr_decay_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0005

        # *** Scheduler
        self.scheduler = 'cosine'

    def set_network(self):
        self.model = 'T_v0'
        self.backbone = 'vgg16bn'
        self.ckpt = None

    def set_training_opts(self):
        # *** Print Option
        self.val_freq = 1
        self.print_freq = 50

        # *** Training
        self.batch_size = 16
        self.num_workers = 1
        self.epochs = 100

        # *** Save option
        self.save_freq = 10
        self.wandb = False

    def set_test_opts(self):
        self.ckpt = None

