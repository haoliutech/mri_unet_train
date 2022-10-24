from pathlib import Path
import re
import os
import torch
import torchio as tio
from utils.utils import get_research_groups_dict

class TioData(object):
    

    def __init__(self, config=None, apply_transforms=True, verbose=True):
                            
        config = self._add_config_defaults(config={})

        self.random_split = config['random_split']

        self.verbose = verbose
        if apply_transforms:
            self._build_transforms(config)
        else:
            self.training_transform = None
            self.validation_transform = None

        return None

    def _add_config_defaults(self, config):
        cwd_path = directory = Path(os.getcwd())
        parent_path = cwd_path.parent.absolute()
        data_path = os.path.join(parent_path,"data/data_organized_1year")

       # DEFAULT VALUES OF CONFIG DICT
        config_defaults =  {
                            "CropOrPad": {"target_shape": [224, 256, 1]},
                            "Normalization": {"RescaleIntensity": {"out_min_max": [-1, 1], "percentiles": [1, 99]}},
                            "RandomAffine": None,
                            "RandomBiasField": None,
                            "RandomBlur": None,
                            "RandomElasticDeformation": None,
                            "RandomGhosting": None,
                            "RandomMotion": None,
                            "SkipAugmentationProbability": None,
                            "batch_size": 1,
                            "dataset_dir": data_path,

                            "history_dir": "/home/apollo/data/history",
                            "include_T1": False,
                            "random_split": False,
                            "valid_batch_size": 1,
                            "verbose": False,
                            "workers": 4,
                            "device" : "cpu",
                            #Hao's editing
                            "epochs": 10,                            
                            "early_stopping": 50,
                            "callback_log_images_freq": 100,
                            "callback_log_model_freq": 1,   
                            "loss_criterion": {"loss_function": "DiceLoss"},
                            "optimizer": "sgd",     
                            "learning_rate": 0.001, 
                            "log_dce_loss": 10,              
                            "min_val_loss": 0.25,     
                           }
        for k, v in config_defaults.items():
            if  k not in config:
                config[k] = config_defaults[k]

        return config



    def _build_transforms(self, config):

        config = self._add_config_defaults(config)

        training_composition = []
        validation_composition = []
        if config['CropOrPad']:
            kwargs = config['CropOrPad']
            training_composition.append(tio.CropOrPad(**kwargs))
            validation_composition.append(tio.CropOrPad(**kwargs))

        if config['Normalization']:
            # This transform must be a selection, if RandomNoise is selected it must be combined with ZNormalization
            tio_transform = {'ZNormalization' : tio.ZNormalization, 
                             'RandomNoise' : tio.RandomNoise,
                             'RescaleIntensity' : tio.RescaleIntensity}

            for transform_name, kwargs in config['Normalization'].items():
                for key, val in kwargs.items():
                    # Hack : Need to figure out how to pass callable through wandb as it is setting the callable to empty string
                    if val == 'tio.ZNormalization.mean':
                        kwargs[key] = tio.ZNormalization.mean

                training_composition.append(tio_transform[transform_name](**kwargs))
                if transform_name in ['ZNormalization', 'RescaleIntensity']:
                    validation_composition.append(tio_transform[transform_name](**kwargs))
                
        if config['RandomAffine']:
            kwargs = config['RandomAffine']
            training_composition.append(tio.RandomAffine(**kwargs))
            #validation_composition.append(tio.RandomAffine(**kwargs))

        if config['RandomElasticDeformation']:
            kwargs = config['RandomElasticDeformation']
            training_composition.append(tio.RandomElasticDeformation(**kwargs))
            #validation_composition.append(tio.RandomElasticDeformation(**kwargs))

        if config['RandomBlur']:
            kwargs = config['RandomBlur']
            training_composition.append(tio.RandomBlur(**kwargs))
            #validation_composition.append(tio.RandomBlur(**kwargs))

        if config['RandomBiasField']:
            kwargs = config['RandomBiasField']
            training_composition.append(tio.RandomBiasField(**kwargs))
            #validation_composition.append(tio.RandomBiasField(**kwargs))

        if config['RandomGhosting']:
            kwargs = config['RandomGhosting']
            training_composition.append(tio.RandomGhosting(**kwargs))
            #validation_composition.append(tio.RandomGhosting(**kwargs))

        if config['RandomMotion']:
            kwargs = config['RandomMotion']
            training_composition.append(tio.RandomMotion(**kwargs))
            #validation_composition.append(tio.RandomMotion(**kwargs))
                      
        if config['SkipAugmentationProbability']:
            # With probability p we will skip augmentation and show the network 
            # images with only basic preprocessing (ie., same as validation)
            p = float(config['SkipAugmentationProbability'])

            oneof_dict = {
                tio.Compose(validation_composition): p,
                tio.Compose(training_composition): 1-p
                }  

            self.training_transform = tio.OneOf(oneof_dict)

        else:
            self.training_transform = tio.Compose(training_composition)

        self.validation_transform = tio.Compose(validation_composition)

    def _tio_subjects(self, config):

        config = self._add_config_defaults(config)
   
        if config['include_T1'] == True:
            # If include_T1 is selected, then a second folder images_T1 must be included in the dataset_dir
            # Inputs will be converted to a 2-channel input problem
            if self.verbose: print('include_T1=True, creating 2C dataset')

            return self._tio_subjects_T2_and_T1(config)
        else:

            if self.verbose: print('include_T1=False, creating 1C dataset')

            return self._tio_subjects_T2_only(config)

    def _get_patient_id_from_image_path(self, image_path):
        # Helper function ussed in _tio_subjects_***()

        image_path=str(image_path)

        regex_1 = re.search(r'(?<=PPMI_)(.*)(?=_MR)', image_path)
        regex_2 = re.search(r'(?<=image)_(.*?)_(.*?).nii', image_path)
        if regex_1:
            patient_id = regex_1.group(1)
        elif regex_2:
            patient_id =  regex_2.group(1)
        else:
            raise ValueError(f'Error: patient id could not be extracted using regex from {image_path}')

        return patient_id

    def _tio_subjects_T2_only(self, config):

        # Dataset
        dataset_dir = Path(config['dataset_dir'])

        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        image_paths = sorted(images_dir.glob('*.nii'))
        label_paths = sorted(labels_dir.glob('*.nii.gz'))

        assert len(image_paths) == len(label_paths)

        subjects = []
        for (image_path, label_path) in zip(image_paths, label_paths):

            patient_id = self._get_patient_id_from_image_path(image_path)

            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
                patient_id = patient_id,
            )
            subjects.append(subject)

        return subjects


    def _tio_subjects_T2_and_T1(self, config):

        # Dataset
        dataset_dir = Path(config['dataset_dir'])

        images_dir = dataset_dir / 'images'
        images_T1_dir = dataset_dir / 'images_T1'
        labels_dir = dataset_dir / 'labels'
        image_paths = sorted(images_dir.glob('*.nii'))
        image_T1_paths = sorted(images_T1_dir.glob('*.nii'))
        label_paths = sorted(labels_dir.glob('*.nii.gz'))
        assert len(image_paths) == len(label_paths)
        assert len(image_paths) == len(image_T1_paths)

        subjects = []
        for (image_path, image_T1_path, label_path) in zip(image_paths, image_T1_paths, label_paths):

            patient_id = self._get_patient_id_from_image_path(image_path)
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                image_T1=tio.ScalarImage(image_T1_path),
                label=tio.LabelMap(label_path),
                patient_id = patient_id,
            )
            subjects.append(subject)

        return subjects

    def _get_patient_group_from_id(self, patient_id, research_groups):
        
        if int(patient_id) not in research_groups.keys():
            print(f'WARNING: research group for patient_id: {patient_id} not found, omitting this patient from dataset')
            return None
        
        else:
            return research_groups[int(patient_id)]


    def _tio_subjects_T2_and_T1_run_as_classifier(self, config):

        research_groups = get_research_groups_dict()

        # Dataset
        dataset_dir = Path(config['dataset_dir'])

        images_dir = dataset_dir / 'images'
        images_T1_dir = dataset_dir / 'images_T1'
        labels_dir = dataset_dir / 'labels'
        image_paths = sorted(images_dir.glob('*.nii'))
        image_T1_paths = sorted(images_T1_dir.glob('*.nii'))
        label_paths = sorted(labels_dir.glob('*.nii.gz'))
        assert len(image_paths) == len(label_paths)
        assert len(image_paths) == len(image_T1_paths)

        subjects = []
        for (image_path, image_T1_path, label_path) in zip(image_paths, image_T1_paths, label_paths):

            patient_id = self._get_patient_id_from_image_path(image_path)
            research_group = self._get_patient_group_from_id(patient_id, research_groups)
            if research_group is not None:
                subject = tio.Subject(
                    image=tio.ScalarImage(image_path),
                    image_T1=tio.ScalarImage(image_T1_path),
                    label=tio.LabelMap(label_path),
                    patient_id = patient_id,
                    research_group = research_group
                )
                subjects.append(subject)

        return subjects

    def _tio_subjects_T2_original_code(self, config):

        # Dataset
        dataset_dir = Path(config['dataset_dir'])

        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        image_paths = sorted(images_dir.glob('*.nii'))
        label_paths = sorted(labels_dir.glob('*.nii.gz'))
        assert len(image_paths) == len(label_paths)

        regex_compile = re.compile("(?<=PPMI_)(.*)(?=_MR)")
        subjects = []
        for (image_path, label_path) in zip(image_paths, label_paths):

            patient_id = regex_compile.search(str(image_path)).group(1)
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
                patient_id = patient_id,
            )
            subjects.append(subject)

        return subjects


    def _tio_datasets(self, config):    

        config = self._add_config_defaults(config)

        subjects = self._tio_subjects(config)

        cwd_path = directory = Path(os.getcwd())
        parent_path = cwd_path.parent.absolute()
        data_path = os.path.join(parent_path,"data/data_organized_1year")
        t2_img_path = os.path.join(data_path,"images/PPMI_3104_MR_Axial_PD-T2_TSE__br_raw_20120503095911833_66_S148995_I301554.nii")
        

        # Hao's code start

        t2_img = tio.ScalarImage(t2_img_path)

        training_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(t2_img),
        ])

        
        dataset = tio.SubjectsDataset(subjects, transform=training_transform)
        subjects = tio.SubjectsDataset(subjects, transform=training_transform)

        # Hao's code end


        
        if self.verbose: print('Dataset size:', len(dataset), 'subjects')

        training_split_ratio = 0.8
        num_subjects = len(dataset)
        num_training_subjects = int(training_split_ratio * num_subjects)
        num_validation_subjects = num_subjects - num_training_subjects

        num_split_subjects = num_training_subjects, num_validation_subjects

        # If we do not want a random split, include 'random_split'=False in config
        if not self.random_split:
            torch.manual_seed(0)
            pass
        training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

        

        validation_transform = tio.Compose([
            tio.ToCanonical(1),
            tio.Resample(4),
        ])

        training_set = tio.SubjectsDataset(
            training_subjects, transform=self.training_transform)

        validation_set = tio.SubjectsDataset(
            validation_subjects, transform=self.validation_transform)

        

        if self.verbose: print('Training set:', len(training_set), 'subjects')
        if self.verbose: print('Validation set:', len(validation_set), 'subjects')

        self.len_train = len(training_set)
        self.len_valid = len(validation_set)

        return training_set, validation_set

    def tio_data_loaders(self, config):


        training_set, validation_set = self._tio_datasets(config)

        training_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['workers'],
        )

        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=config['valid_batch_size'],
            num_workers=config['workers'],
        )

        return training_loader, validation_loader

    def get_lengths(self):
        return self.len_train, self.len_valid
