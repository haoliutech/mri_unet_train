from utils.wandb_analysis_2C_2D import *
from utils.config_utils import read_json, write_json
from utils.results_manager import ResultsManager
from utils.wandb_utils_v2 import get_wandb_config_from_json, update_config_from_df, compute_metrics_by_run
from utils.wandb_utils_v2 import get_compare_labellers_df_v3

tex_base_path='./FinalReport'
results_path = Path('/Users/hao/MRI_Research_Project/mproj7205/data/report_results/')
multi_SN_RN_path = Path('/Users/hao/MRI_Research_Project/mproj7205/data/multi_SN_RN/npy_2d/SN/')
cwd_path = Path('/Users/hao/MRI_Research_Project/mproj7205')


def map_columns(df):

    #df.reset_index(inplace=True, drop=True)
    df_keys = df.keys()

    # if 'RandomAffine' in df_keys:
    #     df['augmentation'] = df['RandomAffine'].isna()
    #     df['augmentation_network_'] = ( df['network_name'] + df['RandomAffine'].isna().astype('str')).replace(replace_dict)

    if 'augmentation' in df_keys:
        replace_dict = {False: 'No augmentation', True : 'Augmentation'}
        df['augmentation_'] = df['augmentation'].replace(replace_dict)

    replace_dict = {'UNetFalse': 'UNet (no augmentation)', 'UNetTrue' : 'UNet (augmentation)',
                    'UNet_2PlusFalse': 'UNet ++ (no augmentation)', 'UNet_2PlusTrue': 'UNet ++ (augmentation)', 
                    'UNet_3PlusFalse': 'UNet 3+ (no augmentation)', 'UNet_3PlusTrue': 'UNet 3+ (augmentation)', 
                    'smp.UnetFalse' : 'UNet (smp) (no augmentation)', 'smp.UnetTrue' : 'UNet (smp) (augmentation)'
                    }

    inlcude_T1_labels=['T2 only', 'T1 and T2']  
    df['include_T1_'] = df['include_T1'].replace({False: inlcude_T1_labels[0], True : inlcude_T1_labels[1]})
    
    replace_dict = {'TverskyLoss': 'Tversky Loss', 
                    'TverskyFocalLoss': 'Tversky Focal Loss', 
                    'BCELoss': 'BCE Loss', 
                    'FocalLoss': 'Focal Loss', 
                    'WeightedDiceLoss': 'Weighted Dice Loss', 
                    'FBetaLoss' : 'F-Beta Loss', 
                    'BCEDiceLoss' : 'BCE Dice Loss',
                    'DiceLoss' : 'Dice Loss'}
    df['loss_function_'] = df['loss_function'].replace(replace_dict)

    replace_dict = {'TverskyLoss': 'Tversky', 
                    'TverskyFocalLoss': 'Tversky Focal', 
                    'BCELoss': 'BCE', 
                    'FocalLoss': 'Focal', 
                    'WeightedDiceLoss': 'Wgt. Dice', 
                    'FBetaLoss' : 'F-Beta', 
                    'BCEDiceLoss' : 'BCE Dice',
                    'DiceLoss' : 'Dice'}
    df['loss_function_trim'] = df['loss_function'].replace(replace_dict)


    replace_dict = {'UNet_2Plus': 'UNet ++', 'UNet_3Plus': 'UNet 3+', 'smp.Unet' : 'UNet (smp)'}
    df['network_name_'] = df['network_name'].replace(replace_dict)


    return df

def append_mean_median(df, df_box):
    df['mean dsc'] = 0
    df['median dsc'] = 0

    for idx in df.index:
        model_name = df.loc[idx]['model_name']
        model_name = df.loc[idx]['model_name']
        dsc_vals = df_box[df_box['model_name']==model_name]['dsc']
        df.loc[idx, 'mean dsc'] = dsc_vals.mean()
        df.loc[idx, 'median dsc'] = dsc_vals.median()

        df.loc[idx, 'mean tp'] = df_box[df_box['model_name']==model_name]['tp'].mean()
        df.loc[idx, 'mean fp'] = df_box[df_box['model_name']==model_name]['fp'].mean()
        df.loc[idx, 'mean fn'] = df_box[df_box['model_name']==model_name]['fn'].mean()

        df.loc[idx, 'mean ppv'] = df_box[df_box['model_name']==model_name]['ppv'].mean()
        df.loc[idx, 'mean sensitivity'] = df_box[df_box['model_name']==model_name]['sensitivity'].mean()
        df.loc[idx, 'mean dscoeff'] = df_box[df_box['model_name']==model_name]['dscoeff'].mean()

    return df

def compute_metrics_by_df(df, load_from_file=False, save_to_file=True):
    
    def ppv(tp, fp):
        return (tp)/(tp+fp)

    def sensitivity(tp, fn):
        return (tp)/(tp+fn)

    def dscoeff(tp, fn, fp):
        return (2*tp)/(2*tp+fn+fp)

    df_box_fname = results_path / 'df_box.csv'

    if load_from_file and df_box_fname.is_file():
        return pd.read_csv(df_box_fname, index_col=0)

    df = map_columns(df)

    df_all = None
    
    for idx in df.index:
        df1 = compute_metrics_by_run(df, idx=idx, sweep_name=None)

        df1['ppv'] = df1.apply(lambda x: ppv(x.tp, x.fp), axis=1)
        df1['sensitivity'] = df1.apply(lambda x: sensitivity(x.tp, x.fn), axis=1)
        df1['dscoeff'] = df1.apply(lambda x: dscoeff(x.tp, x.fn, x.fp), axis=1)

        #df1['include_T1_'] = df['include_T1'].replace({False: inlcude_T1_labels[0], True : inlcude_T1_labels[1]})
        df1['include_T1_'] = df.loc[idx]['include_T1_']
        df1['include_T1'] = df.loc[idx]['include_T1']
        df1['loss_function'] = df.loc[idx]['loss_function']
        df1['loss_function_'] = df.loc[idx]['loss_function_']
        df1['network_name'] = df.loc[idx]['network_name']
        df1['network_name_'] = df.loc[idx]['network_name_']
        df1['name'] = df.loc[idx]['name']
        df1['table_name'] = df.loc[idx]['name']
        if 'pre_trained_' in df.keys():
            df1['pre_trained_'] = df.loc[idx]['pre_trained_']
        if 'augmentation' in df.keys():
            df1['augmentation'] = df.loc[idx]['augmentation']
            df1['augmentation_'] = df.loc[idx]['augmentation_']

        if 'beta' in df.keys():
            df1['beta'] = df.loc[idx]['beta']

        if 'loss_function_trim' in df.keys():
            df1['loss_function_trim'] = df.loc[idx]['loss_function_trim']

        if isinstance(df_all, pd.DataFrame):
            df_all = pd.concat([df_all, df1], axis=0, ignore_index=True)
        else:
            df_all = df1
    #df_all =  pd.concat([df_all, df], axis=1, ignore_index=True)

    if save_to_file:         
        df_all.to_csv(df_box_fname, index=True)

    return df_all

def create_baseline_images(patent_ids, 
                           crop_shape=None):
    """
    patient_ids: list of patient ids. If not set, all examples with two labels will be printed.

    """
    # figs_save_path = results_path / tex_base_path /'figs'
    # figs_save_path.mkdir(parents=True, exist_ok=True)

    # tex_save_path = results_path / tex_base_path /'tex'
    # tex_save_path.mkdir(parents=True, exist_ok=True)

    npy_path_0 = multi_SN_RN_path / "0/angus/"
    npy_path_1 = multi_SN_RN_path / "1/ben/"

    npy_1 = os.listdir(npy_path_0 / 'labels')
    npy_2 = os.listdir(npy_path_1 / 'labels')

    both = [x for x in npy_1 if x in npy_2]

    # Remove 3607 from this list
    both = [x for x in both if '3828' not in str(x)]
    both = [x for x in both if '3607' not in str(x)]

    pid2label_dict = {}
    for i, k in enumerate(both):
        regex = re.search(r'label_(.*)_(.*)', k)
        if regex:
            patient_id = int(regex.group(1))
            pid2label_dict[patient_id] = i

    wa = WandbAnalysis() 

    patent_ids = [x for x in patent_ids if x in pid2label_dict.keys()]

    # Get the index from patient ids
    index_list = [pid2label_dict[i] for i in patent_ids]

    for label_fname in both:

        label_ = npy_path_0 / 'labels' / label_fname
        prediction_ = npy_path_1 / 'labels' /  label_fname
        original_ = npy_path_0 / 'images' /  label_fname.replace('label', 'image')

        title_prefix=''
        model_name=''

        regex = re.search(r'label_(.*)_(.*)', label_fname)
        if regex:
            patient_id = int(regex.group(1))
        else:
            print(f'patient id not found for {label_fname}')
            continue

        if patient_id not in patent_ids:
            continue

        if label_.is_file() and prediction_.is_file() and original_.is_file():
            
            label_ = torch.from_numpy(np.load(label_))
            prediction_ = torch.from_numpy(np.load(prediction_))
            original_ = torch.from_numpy(np.load(original_))

            f, ax = plt.subplots(1,1, figsize=(16,10))
            metrics_ = wa.plot_results_from_prediction(original_=original_, 
                                                       label_=label_, 
                                                       prediction_=prediction_,
                                                       ax=ax,
                                                       show_title=False,
                                                       title_prefix_='',
                                                       model_name_ = model_name,
                                                       patient_id_ = patient_id,
                                                       return_metrics=True,
                                                       crop_shape=crop_shape, 
                                                       zoom_inset=2.5)

            if metrics_ is None:
                plt.close()
                continue

            split_caption = metrics_.replace(' ', '').replace(':',',').split(',')

            save_path = results_path / tex_base_path / 'images' / str(patient_id) 
            save_path.mkdir(parents=True, exist_ok=True)
            png_fname = save_path /  'baseline.png'
            f.savefig(png_fname, bbox_inches='tight')
            plt.close()

            metrics = dict(zip(split_caption[0::2], split_caption[1::2]))

            subcaption = f"DSC={metrics['DSC']} ({metrics['TP']}/{metrics['FN']}/{metrics['FP']})"

            tex_fname = save_path / f'{png_fname.stem}.txt'    
            f = open(tex_fname, "w")
            f.write(subcaption)
            f.close()   

            write_to_input_commands_tex(tex_fname)



def create_reference_images(patent_ids, crop_shape=None):
    """
    patient_ids: list of patient ids. If not set, all examples with two labels will be printed.

    """
    # figs_save_path = results_path / tex_base_path /'figs'
    # figs_save_path.mkdir(parents=True, exist_ok=True)

    # tex_save_path = results_path / tex_base_path /'tex'
    # tex_save_path.mkdir(parents=True, exist_ok=True)

    npy_path_0 = multi_SN_RN_path / "0/angus/"
    npy_path_1 = multi_SN_RN_path / "1/ben/"

    npy_1 = os.listdir(npy_path_0 / 'labels')
    npy_2 = os.listdir(npy_path_1 / 'labels')

    both = [x for x in npy_1 if x in npy_2]

    # Remove 3607 from this list
    both = [x for x in both if '3828' not in str(x)]
    both = [x for x in both if '3607' not in str(x)]

    pid2label_dict = {}
    for i, k in enumerate(both):
        regex = re.search(r'label_(.*)_(.*)', k)
        if regex:
            patient_id = int(regex.group(1))
            pid2label_dict[patient_id] = i

    wa = WandbAnalysis() 

    patent_ids = [x for x in patent_ids if x in pid2label_dict.keys()]

    # Get the index from patient ids
    index_list = [pid2label_dict[i] for i in patent_ids]

    for label_fname in both:

        label_ = npy_path_0 / 'labels' / label_fname
        prediction_ = npy_path_1 / 'labels' /  label_fname
        original_ = npy_path_0 / 'images' /  label_fname.replace('label', 'image')
        original_T1_ = npy_path_0 / 'images_T1' /  label_fname.replace('label', 'image_T1')

        title_prefix=''
        model_name=''

        regex = re.search(r'label_(.*)_(.*)', label_fname)
        if regex:
            patient_id = int(regex.group(1))
        else:
            print(f'patient id not found for {label_fname}')
            continue

        if patient_id not in patent_ids:
            continue

        if label_.is_file() and prediction_.is_file() and original_.is_file() and original_T1_.is_file():
            

            label_ = torch.from_numpy(np.load(label_))
            prediction_ = torch.from_numpy(np.load(prediction_))
            original_ = torch.from_numpy(np.load(original_))
            original_T1_ = torch.from_numpy(np.load(original_T1_))
            zeros_ = torch.zeros(label_.shape)

            overwrite_window = wa.get_cropped_window_dims(np.rot90(wa.crop_npy_to_shape(prediction_)+wa.crop_npy_to_shape(label_)))

            f, ax = plt.subplots(1,1, figsize=(16,10))
            _ = wa.plot_results_from_prediction(original_=original_, 
                                                label_=zeros_, 
                                                prediction_=zeros_,
                                                ax=ax,
                                                show_title=False,
                                                title_prefix_='',
                                                model_name_ = model_name,
                                                patient_id_ = patient_id,
                                                return_metrics=False,
                                                crop_shape=crop_shape, 
                                                overwrite_window=overwrite_window,
                                                hide_legend=True,
                                                zoom_inset=2.5)

            save_path = results_path / tex_base_path / 'images' / str(patient_id) 
            save_path.mkdir(parents=True, exist_ok=True)
            png_fname = save_path /  'T2_only.png'
            f.savefig(png_fname, bbox_inches='tight')
            plt.close()

            f, ax = plt.subplots(1,1, figsize=(16,10))
            _ = wa.plot_results_from_prediction(original_=original_T1_, 
                                                label_=zeros_, 
                                                prediction_=zeros_,
                                                ax=ax,
                                                show_title=False,
                                                title_prefix_='',
                                                model_name_ = model_name,
                                                patient_id_ = patient_id,
                                                return_metrics=False,
                                                crop_shape=crop_shape, 
                                                overwrite_window=overwrite_window,
                                                hide_legend=True,
                                                zoom_inset=2.5)

            save_path = results_path / tex_base_path / 'images' / str(patient_id) 
            save_path.mkdir(parents=True, exist_ok=True)
            png_fname = save_path /  'T2_T1.png'
            f.savefig(png_fname, bbox_inches='tight')
            plt.close()

            f, ax = plt.subplots(1,1, figsize=(16,10))
            _ = wa.plot_results_from_prediction(original_=original_, 
                                                label_=label_, 
                                                prediction_=zeros_,
                                                ax=ax,
                                                show_title=False,
                                                title_prefix_='',
                                                model_name_ = model_name,
                                                patient_id_ = patient_id,
                                                return_metrics=False,
                                                crop_shape=crop_shape, 
                                                overwrite_window=overwrite_window,
                                                hide_legend=True,
                                                zoom_inset=2.5)

            save_path = results_path / tex_base_path / 'images' / str(patient_id) 
            save_path.mkdir(parents=True, exist_ok=True)
            png_fname = save_path /  'labeller1.png'
            f.savefig(png_fname, bbox_inches='tight')
            plt.close()


            f, ax = plt.subplots(1,1, figsize=(16,10))
            _ = wa.plot_results_from_prediction(original_=original_, 
                                                label_=zeros_, 
                                                prediction_=prediction_,
                                                ax=ax,
                                                show_title=False,
                                                title_prefix_='',
                                                model_name_ = model_name,
                                                patient_id_ = patient_id,
                                                return_metrics=False,
                                                crop_shape=crop_shape, 
                                                overwrite_window=overwrite_window,
                                                hide_legend=True,
                                                zoom_inset=2.5)

            save_path = results_path / tex_base_path / 'images' / str(patient_id) 
            save_path.mkdir(parents=True, exist_ok=True)
            png_fname = save_path /  'labeller2.png'
            f.savefig(png_fname, bbox_inches='tight')
            plt.close()


def create_experiment_images(df, 
                             patent_ids, 
                             crop_shape=None,
                             verbose=False):
    """
    patient_ids: list of patient ids. If not set, all examples with two labels will be printed.

    """
 
    for idx in df.index:
        project_name = df.loc[idx]['project_name']
        model_name = df.loc[idx]['model_name'].lower()
        network_name = df.loc[idx]['network_name']
        experiment_name = model_name.replace('.pt', '').replace('-', '_')

        wa = WandbAnalysis(project_name=project_name) 

        config_fname = cwd_path / f"configs/{project_name}.json"
        config = get_wandb_config_from_json(config_fname)
        config = update_config_from_df(config, df_series = df.loc[idx])  
        config['batch_size']=1
        config['valid_batch_size']=1
        config = Bunch(config)

        network = load_model(config)

        device = config.device

        if 'include_T1' in config.keys():
            include_T1 = config.include_T1
        else:
            include_T1 = False        
        
        _, val_loader = build_datasets(config, verbose=verbose)

        network.float()
        is_train=False
        use_sigmoid = get_use_sigmoid(config)
        use_sigmoid = False

        for _, data in enumerate(val_loader):

            patient_id = int(data['patient_id'][0])

            if int(patient_id) == 3607 or int(patient_id) == 3828:
                continue

            inputs = data['image'][tio.DATA]
            inputs = np.squeeze(inputs, axis=4)

            if include_T1:
                inputs_T1 = data['image_T1'][tio.DATA]
                inputs_T1 = np.squeeze(inputs_T1, axis=4)
                inputs = torch.cat((inputs, inputs_T1), 1)

            inputs = inputs.to(device)

            labels = data['label'][tio.DATA]
            labels = np.squeeze(labels, axis=4)
            labels = labels.to(device)

            # Forward pass
            with torch.set_grad_enabled(is_train):
                if use_sigmoid:
                    predictions = torch.sigmoid(network(inputs))
                else:
                    predictions = network(inputs)
                    
                # model_name_check = model_name
                # model_name = config.model_name
                # print(model_name_check, model_name)

                inputs = inputs[:,0,:,:].unsqueeze(1)

                f, ax = plt.subplots(1,1, figsize=(16,10))
                try:
                    metrics_ = wa.plot_results_from_prediction(original_=inputs, 
                                                                label_=labels, 
                                                                prediction_=predictions,
                                                                ax=ax,
                                                                show_title=False,
                                                                title_prefix_='',
                                                                model_name_ = model_name,
                                                                patient_id_ = patient_id,
                                                                return_metrics=True,
                                                                crop_shape=crop_shape, 
                                                                zoom_inset=2.5)
                except:
                    plt.close()
                    continue

                if metrics_ is None:
                    plt.close()
                    continue

                split_caption = metrics_.replace(' ', '').replace(':',',').split(',')

                if float(split_caption[-1]) < 0.0:
                    plt.close()
                    continue

                save_path = results_path / tex_base_path / 'images' / str(patient_id) 
                save_path.mkdir(parents=True, exist_ok=True)
                png_fname = save_path /  f'{experiment_name}.png'
                f.savefig(png_fname, bbox_inches='tight')
                plt.close()

                metrics = dict(zip(split_caption[0::2], split_caption[1::2]))

                subcaption = f"DSC={metrics['DSC']} ({metrics['TP']}/{metrics['FN']}/{metrics['FP']})"

                tex_fname = save_path / f'{png_fname.stem}.txt'    
                f = open(tex_fname, "w")
                f.write(subcaption)
                f.close()   

                write_to_input_commands_tex(tex_fname)

def get_caption(tex_fname):
    f = open(tex_fname, "r")
    line = f.read()
    f.close()   
    return line


def create_3col_baseline_figure(patient_ids, 
                                fname_stem = 'compare_labellers_baseline_9x9',
                                caption = '',
                                short_caption=None,
                                verbose=True):

    latex_str = r"""
\begin{figure}[!ht]
    \centering
    \begin{tabular}{cccc}
    \rotatebox[origin=c]{90}{\textbf{RHEADING0}}  & ADJUSTBOX00 & ADJUSTBOX01 & ADJUSTBOX02 \\[-2mm]
    & SUBCAPTION00 & SUBCAPTION01 & SUBCAPTION02 \\[2mm]
    \rotatebox[origin=c]{90}{\textbf{RHEADING1}}  & ADJUSTBOX10 & ADJUSTBOX11 & ADJUSTBOX12 \\[-2mm]
    & SUBCAPTION10 & SUBCAPTION11 & SUBCAPTION12 \\[2mm]
    \rotatebox[origin=c]{90}{\textbf{RHEADING2}}  & ADJUSTBOX20 & ADJUSTBOX21 & ADJUSTBOX22 \\[-2mm]
    & SUBCAPTION20 & SUBCAPTION21 & SUBCAPTION22 \\[2mm]
    \end{tabular}
    \caption[SHORTCAPTION]{FULLCAPTION}
    \label{fig:LABEL}
\end{figure}
"""

    # NO NEED FOR THIS HEADER ROW?
    # & \textbf{CHEADING0} & \textbf{CHEADING1} & \textbf{CHEADING2} \\[2mm]   
    #CHEADINGS = ['Baseline', 'T2 Only', 'T1 and T2'], 
    #replace_dict = {'CHEADING0' : CHEADINGS[0], 'CHEADING1' : CHEADINGS[1], 'CHEADING2' : CHEADINGS[2], 'FULLCAPTION' : CAPTION, 'LABEL' : fig_fname.stem}
    
    figs_path = results_path / f"{tex_base_path}/figs"
    figs_path.mkdir(parents=True, exist_ok=True)
    fig_fname = figs_path/ f'{fname_stem}.tex'

    ADJUSTBOX_ = lambda x: r"""\adjustbox{valign=m,vspace=1pt}{\includegraphics[width=.29\linewidth]{FILENAME}}""".replace('FILENAME', x)

    replace_dict = {'RHEADING0' : "Most Agreement", 'RHEADING1' : "Medium Agreement", 'RHEADING2' : "Least Agreement", 'FULLCAPTION' : caption, 'LABEL' : fig_fname.stem}

    if short_caption:
        short_caption_ = f'[{short_caption}]'
    else:
        short_caption_ = ''

    replace_dict.update({'[SHORTCAPTION]' : short_caption_})


    assert len(patient_ids) >= 9
    median_ = len(patient_ids)//2
    patient_ids = patient_ids[-3:] + patient_ids[median_-2:median_+1] + patient_ids[:3] 

    p_idx = 0
    for row_idx in range(3):
        for col_idx in range(3):

            patient_id = patient_ids[p_idx]
            p_idx+=1

            latex_path = f"{tex_base_path}/images/{str(patient_id)}"
        
            replace_dict[f'ADJUSTBOX{row_idx}{col_idx}'] =  ADJUSTBOX_(f'{latex_path}/baseline.png')
            replace_dict[f'SUBCAPTION{row_idx}{col_idx}'] =  get_caption(results_path / f'{latex_path}/baseline.txt')

    for key, value in replace_dict.items():
        latex_str = latex_str.replace(key, value)

    f = open(fig_fname, "w")
    f.write(latex_str)
    f.close()   

    write_to_input_commands_tex(fig_fname)

    if verbose: print(latex_str)


def create_3col_reference_figure(patient_ids,  
                                 fname_stem = 'compare_labellers_reference_9x9',
                                 caption = '',
                                 short_caption=None,
                                 verbose=True):

    latex_str = r"""
\begin{figure}[!ht]
    \centering
    \begin{tabular}{cccc}
    & \textbf{CHEADING0} & \textbf{CHEADING1} & \textbf{CHEADING2} \\[2mm]   
    \rotatebox[origin=c]{90}{\textbf{RHEADING0}}  & ADJUSTBOX00 & ADJUSTBOX01 & ADJUSTBOX02 \\[-2mm]
    \rotatebox[origin=c]{90}{\textbf{RHEADING1}}  & ADJUSTBOX10 & ADJUSTBOX11 & ADJUSTBOX12 \\[-2mm]
    \rotatebox[origin=c]{90}{\textbf{RHEADING2}}  & ADJUSTBOX20 & ADJUSTBOX21 & ADJUSTBOX22 \\[-2mm]
    \end{tabular}
    \caption[SHORTCAPTION]{FULLCAPTION}
    \label{fig:LABEL}
\end{figure}
"""
    # NO NEED FOR THIS?
    # & SUBCAPTION00 & SUBCAPTION01 & SUBCAPTION02 \\[2mm]
    # & SUBCAPTION10 & SUBCAPTION11 & SUBCAPTION12 \\[2mm]
    # & SUBCAPTION20 & SUBCAPTION21 & SUBCAPTION22 \\[2mm]

    CHEADINGS = ['T2', 'T1 (coregistered)', 'Compare labellers'] 
    
    figs_path = results_path / f"{tex_base_path}/figs"
    figs_path.mkdir(parents=True, exist_ok=True)
    fig_fname = figs_path/ f'{fname_stem}.tex'

    ADJUSTBOX_ = lambda x: r"""\adjustbox{valign=m,vspace=1pt}{\includegraphics[width=.29\linewidth]{FILENAME}}""".replace('FILENAME', x)

    replace_dict = {'CHEADING0' : CHEADINGS[0], 'CHEADING1' : CHEADINGS[1], 'CHEADING2' : CHEADINGS[2], \
                    'FULLCAPTION' : caption, 'LABEL' : fig_fname.stem}

    if short_caption:
        short_caption_ = f'[{short_caption}]'
    else:
        short_caption_ = ''

    replace_dict.update({'[SHORTCAPTION]' : short_caption_})

                    # 'RHEADING0' : "Most Agreement", 'RHEADING1' : "Median Agreement", 'RHEADING2' : "Least Agreement", \

    assert len(patient_ids) >= 3
    median_ = len(patient_ids)//2
    patient_ids = [patient_ids[-1], patient_ids[median_], patient_ids[0]]

    p_idx = 0
    for row_idx in range(3):
        patient_id = patient_ids[p_idx]
        p_idx+=1

        latex_path = f"{tex_base_path}/images/{str(patient_id)}"


        replace_dict[f'ADJUSTBOX{row_idx}{0}'] =  ADJUSTBOX_(f'{latex_path}/T2_only.png')
        #replace_dict[f'SUBCAPTION{row_idx}{1}'] =  get_caption(results_path / f'{latex_path}/baseline.txt')

        replace_dict[f'ADJUSTBOX{row_idx}{1}'] =  ADJUSTBOX_(f'{latex_path}/T2_T1.png')
        #replace_dict[f'SUBCAPTION{row_idx}{0}'] =  get_caption(results_path / f'{latex_path}/baseline.txt')
        
        replace_dict[f'ADJUSTBOX{row_idx}{2}'] =  ADJUSTBOX_(f'{latex_path}/baseline.png')
        #replace_dict[f'SUBCAPTION{row_idx}{2}'] =  get_caption(results_path / f'{latex_path}/baseline.txt')

        replace_dict[f'RHEADING{row_idx}'] = f'Subject {str(patient_id)}'

        row_idx += 1

    for key, value in replace_dict.items():
        latex_str = latex_str.replace(key, value)

    if 'ADJUSTBOX' in latex_str:
        latex_str=""

    f = open(fig_fname, "w")
    f.write(latex_str)
    f.close()   

    write_to_input_commands_tex(fig_fname)

    if verbose: print(latex_str)


def create_3col_figure_v0(df, 
                          patient_id, 
                          fname_stem,
                          CHEADINGS = ['Baseline', 'T2 Only', 'T1 and T2'], 
                          caption = '',
                          short_caption=None,
                          hue_column = 'include_T1_',
                          sort_column = 'loss_function_',
                          verbose=True):

    latex_str = r"""
\begin{figure}[!ht]
    \centering
    \begin{tabular}{cccc}
    & \textbf{CHEADING0} & \textbf{CHEADING1} & \textbf{CHEADING2} \\[2mm]   
    \rotatebox[origin=c]{90}{\textbf{RHEADING0}}  & ADJUSTBOX00 & ADJUSTBOX01 & ADJUSTBOX02 \\[-2mm]
    & SUBCAPTION00 & SUBCAPTION01 & SUBCAPTION02 \\[2mm]
    \rotatebox[origin=c]{90}{\textbf{RHEADING1}}  & ADJUSTBOX10 & ADJUSTBOX11 & ADJUSTBOX12 \\[-2mm]
    & SUBCAPTION10 & SUBCAPTION11 & SUBCAPTION12 \\[2mm]
    \rotatebox[origin=c]{90}{\textbf{RHEADING2}}  & ADJUSTBOX20 & ADJUSTBOX21 & ADJUSTBOX22 \\[-2mm]
    & SUBCAPTION20 & SUBCAPTION21 & SUBCAPTION22 \\[2mm]
    \end{tabular}
    \caption[SHORTCAPTION]{FULLCAPTION}
    \label{fig:LABEL}
\end{figure}
"""

    ADJUSTBOX_ = lambda x: r"""\adjustbox{valign=m,vspace=1pt}{\includegraphics[width=.29\linewidth]{FILENAME}}""".replace('FILENAME', x)

    latex_path = f"{tex_base_path}/images/{str(patient_id)}"

    figs_path = results_path / f"{tex_base_path}/figs/{str(patient_id)}"
    figs_path.mkdir(parents=True, exist_ok=True)
    fig_fname = figs_path/ f'{fname_stem}_3col_{str(patient_id)}.tex'

    replace_dict = {'CHEADING0' : CHEADINGS[0], 'CHEADING1' : CHEADINGS[1], 'CHEADING2' : CHEADINGS[2], 'FULLCAPTION' : caption, 'LABEL' : fig_fname.stem}

    if short_caption:
        short_caption_ = f'[{short_caption}]'
    else:
        short_caption_ = ''

    replace_dict.update({'[SHORTCAPTION]' : short_caption_})


    hue_columns = list(set(df[hue_column]))
    df_grouped = df.sort_values(sort_column)
    row_idx = 0
    for group_name, df_group in df_grouped.groupby(sort_column):
        assert df_group.shape[0] == 2
        
        model_namec1 = df_group[df_group[hue_column] == hue_columns[0]]['model_name'].values[0]
        experiment_namec1 = model_namec1.replace('.pt', '').replace('-', '_')

        model_namec2 = df_group[df_group[hue_column] == hue_columns[1]]['model_name'].values[0]
        experiment_namec2 = model_namec2.replace('.pt', '').replace('-', '_')

        row_name = df_group[sort_column].values[0]
        
        if row_idx == 0:
            print(results_path, f'{latex_path}/baseline.txt')
            replace_dict[f'ADJUSTBOX{row_idx}{0}'] =  ADJUSTBOX_(f'{latex_path}/baseline.png')
            replace_dict[f'SUBCAPTION{row_idx}{0}'] =  get_caption(results_path / f'{latex_path}/baseline.txt')
        else:
            replace_dict[f'ADJUSTBOX{row_idx}{0}'] =  ''
            replace_dict[f'SUBCAPTION{row_idx}{0}'] =  ''
        replace_dict[f'ADJUSTBOX{row_idx}{1}'] =  ADJUSTBOX_(f'{latex_path}/{experiment_namec1}.png')
        replace_dict[f'SUBCAPTION{row_idx}{1}'] =  get_caption(results_path / f'{latex_path}/{experiment_namec1}.txt')

        replace_dict[f'ADJUSTBOX{row_idx}{2}'] =  ADJUSTBOX_(f'{latex_path}/{experiment_namec2}.png')
        replace_dict[f'SUBCAPTION{row_idx}{2}'] =  get_caption(results_path / f'{latex_path}/{experiment_namec2}.txt')

        replace_dict[f'RHEADING{row_idx}'] = row_name
        row_idx += 1

    for key, value in replace_dict.items():
        latex_str = latex_str.replace(key, value)

    if 'ADJUSTBOX' in latex_str:
        latex_str=""

    f = open(fig_fname, "w")
    f.write(latex_str)
    f.close()   

    write_to_input_commands_tex(fig_fname)

    if verbose: print(latex_str)



def create_3col_figure(df, 
                       patient_id, 
                       fname_stem,
                       CHEADINGS = ['T2 Only', 'T1 and T2', 'Baseline'], 
                       caption = '',
                       short_caption=None,
                       hue_column = 'include_T1_',
                       sort_column = 'loss_function_',
                       verbose=True):

    latex_str = r"""
\begin{figure}[!ht]
    \centering
    \begin{tabular}{cccc}
    & \textbf{CHEADING0} & \textbf{CHEADING1} & \textbf{CHEADING2} \\[2mm]   
    \rotatebox[origin=c]{90}{\textbf{RHEADING0}}  & ADJUSTBOX00 & ADJUSTBOX01 & ADJUSTBOX02 \\[-2mm]
    & SUBCAPTION00 & SUBCAPTION01 & SUBCAPTION02 \\[2mm]
    \rotatebox[origin=c]{90}{\textbf{RHEADING1}}  & ADJUSTBOX10 & ADJUSTBOX11 & ADJUSTBOX12 \\[-2mm]
    & SUBCAPTION10 & SUBCAPTION11 & SUBCAPTION12 \\[2mm]
    \rotatebox[origin=c]{90}{\textbf{RHEADING2}}  & ADJUSTBOX20 & ADJUSTBOX21 & ADJUSTBOX22 \\[-2mm]
    & SUBCAPTION20 & SUBCAPTION21 & SUBCAPTION22 \\[2mm]
    \end{tabular}
    \caption[SHORTCAPTION]{FULLCAPTION}
    \label{fig:LABEL}
\end{figure}
"""

    ADJUSTBOX_ = lambda x: r"""\adjustbox{valign=m,vspace=1pt}{\includegraphics[width=.29\linewidth]{FILENAME}}""".replace('FILENAME', x)
    SUPERSCRIPT_ = lambda x: r"""\textsuperscript{(VALUE)}""".replace('VALUE', str(x))

    latex_path = f"{tex_base_path}/images/{str(patient_id)}"

    figs_path = results_path / f"{tex_base_path}/figs/{str(patient_id)}"
    figs_path.mkdir(parents=True, exist_ok=True)
    fig_fname = figs_path/ f'{fname_stem}_3col_{str(patient_id)}.tex'

    replace_dict = {'CHEADING0' : CHEADINGS[0], 'CHEADING1' : CHEADINGS[1], 'CHEADING2' : CHEADINGS[2], 'FULLCAPTION' : caption, 'LABEL' : fig_fname.stem}

    if short_caption:
        short_caption_ = f'[{short_caption}]'
    else:
        short_caption_ = ''

    replace_dict.update({'[SHORTCAPTION]' : short_caption_})


    hue_columns = list(set(df[hue_column]))
    df_grouped = df.sort_values(sort_column)
    row_idx = 0
    for group_name, df_group in df_grouped.groupby(sort_column):
        assert df_group.shape[0] == 2
        
        #model_idx_c0 = df_group[df_group[hue_column] == hue_columns[0]].index[0]
        model_name_c0 = df_group[df_group[hue_column] == hue_columns[0]]['model_name'].values[0]
        experiment_name_c0 = model_name_c0.replace('.pt', '').replace('-', '_')

        #model_idx_c1 = df_group[df_group[hue_column] == hue_columns[1]].index[0]
        model_name_c1 = df_group[df_group[hue_column] == hue_columns[1]]['model_name'].values[0]
        experiment_name_c1 = model_name_c1.replace('.pt', '').replace('-', '_')

        row_name = df_group[sort_column].values[0]

        replace_dict[f'ADJUSTBOX{row_idx}{0}'] =  ADJUSTBOX_(f'{latex_path}/{experiment_name_c0}.png')
        replace_dict[f'SUBCAPTION{row_idx}{0}'] =  get_caption(results_path / f'{latex_path}/{experiment_name_c0}.txt')

        replace_dict[f'ADJUSTBOX{row_idx}{1}'] =  ADJUSTBOX_(f'{latex_path}/{experiment_name_c1}.png')
        replace_dict[f'SUBCAPTION{row_idx}{1}'] =  get_caption(results_path / f'{latex_path}/{experiment_name_c1}.txt')

        if row_idx == 0:
            print(results_path, f'{latex_path}/baseline.txt')
            replace_dict[f'ADJUSTBOX{row_idx}{2}'] =  ADJUSTBOX_(f'{latex_path}/baseline.png')
            replace_dict[f'SUBCAPTION{row_idx}{2}'] =  get_caption(results_path / f'{latex_path}/baseline.txt')
        else:
            replace_dict[f'ADJUSTBOX{row_idx}{2}'] =  ''
            replace_dict[f'SUBCAPTION{row_idx}{2}'] =  ''

        replace_dict[f'RHEADING{row_idx}'] = row_name
        row_idx += 1

    for key, value in replace_dict.items():
        latex_str = latex_str.replace(key, value)

    if 'ADJUSTBOX' in latex_str:
        latex_str=""

    f = open(fig_fname, "w")
    f.write(latex_str)
    f.close()   

    write_to_input_commands_tex(fig_fname)

    if verbose: print(latex_str)



def create_latex_figure(save_path, png_path, caption='', short_caption=None):

    figure_str = r"""\begin{figure}[htp]
    \centering
    \includegraphics[width=0.99\linewidth]{png_path}
    \caption[SHORTCAPTION]{caption_str}
    \label{label_str}
\end{figure}
"""

    if short_caption:
        short_caption_ = f'[{short_caption}]'
    else:
        short_caption_ = ''

    figure_str = figure_str.replace('[SHORTCAPTION]', short_caption_)

    figure_str = figure_str.replace('png_path', png_path).replace('caption_str', caption).replace('label_str',save_path.stem)
 
    f = open(save_path, "w")
    f.write(figure_str)
    f.close()   

    # Also, create the save path for the figure if needed
    save_path = results_path / tex_base_path / 'figs'
    save_path.mkdir(parents=True, exist_ok=True)


def annotate_patches(ax):
    for bar in ax.patches:
        #ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))    
        
        ax.annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')        

def get_column_order(df_box, x_column='loss_function', hue_column='include_T1'):
    
    df1 = df_box.groupby([x_column, hue_column], as_index=False).median().sort_values(['dsc'], ascending=False)
    column_order = list(df1.drop_duplicates(subset=[x_column], keep='first').copy()[x_column])
    return column_order

def filter_df_box(df, df_box, filter_):
    name_list = filter_(df).drop_duplicates(subset=['network_name', 'loss_function', 'include_T1', 'augmentation'], keep='first').copy()['name']
    return df_box[df_box['name'].isin(list(name_list.unique()))]

def plot_from_df(df, 
                 x_column='name', 
                 y_column='dsc', 
                 hue_col='include_T1_', 
                 sns_plot='sns.barplot', 
                 set_order=True,
                 fname_stem=None, 
                 caption='',
                 short_caption=None, 
                 y_column2=None,
                 rotation=None,
                 figsize=(16,6),
                 x_label=None, y_label=None, label_fontsize=24, legend_fontsize=20, \
                 axhline=None, show_annotations=True, set_xlim=None, set_ylim=None):
                 

    if set_order:
        order = get_column_order(df,x_column=x_column, hue_column=hue_col)
    else:
        order = None

    #if sns_plot != 'sns.histplot': df = create_columns_for_results(df)

    #inlcude_T1_labels=['T2 only', 'T1 and T2']    

    sns.set(font_scale=2)

    f, ax = plt.subplots(1,1, figsize=figsize)
    if sns_plot == 'sns.barplot':
        ax = sns.barplot(data=df, x=x_column, y=y_column, hue=hue_col, order=order)

    elif sns_plot == 'sns.violinplot':
        ax = sns.violinplot(data=df, x=x_column, y=y_column, hue=hue_col, split=True, order=order)

    elif sns_plot == 'sns.boxplot':
        ax = sns.boxplot(data=df, x=x_column, y=y_column, hue=hue_col, order=order)

    elif sns_plot == 'sns.histplot':
        ax = sns.histplot(data=df, x=x_column, stat='density', kde=True, bins=10, line_kws={'linewidth':3}, fill=False )

    elif sns_plot == 'sns.lineplot':
        ax = sns.lineplot(data=df, x=x_column, y=y_column, hue=hue_col, ci=None, linewidth = 3)
        if y_column2:
            ax = sns.lineplot(data=df, x=x_column, y=y_column2, hue=hue_col, ci=None)

    elif sns_plot == 'sns.kdeplot':
        ax = sns.kdeplot(data=df, x=x_column, hue=hue_col, kind="kde", fill=False)
        #, cut=0, bw_adjust=1, height=5, aspect=1.6,

    if axhline:
        ax.axhline(axhline, color='r', linestyle='--', linewidth=4)

    if set_xlim :
        ax.set_xbound(set_xlim)
        ax.set_xlim(set_xlim)

    if set_ylim :
        ax.set_ylim(set_ylim)
        

    if x_label:
        ax.set_xlabel(x_label, fontsize=label_fontsize)
    else:
        ax.xaxis.label.set_size(label_fontsize)

    if rotation is not None:
        ax.tick_params(axis='x', rotation=rotation)

    if y_label:
        ax.set_ylabel(y_label, fontsize=label_fontsize)
    else:
        ax.set_ylabel('DSC', fontsize=label_fontsize)

    if show_annotations:
        annotate_patches(ax)
    

    if sns_plot != 'sns.histplot': 
        # plt.legend(title='MRI Channels', loc='upper left', fontsize=legend_fontsize)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=legend_fontsize)

        #plt.legend(title='MRI Channels', loc='upper left', fontsize=legend_fontsize)
        plt.legend(fontsize=legend_fontsize, loc='lower right')


    if fname_stem:

        rel_path = f'{tex_base_path}/images/'
        figs_path = results_path / rel_path
        figs_path.mkdir(parents=True, exist_ok=True)

        fig_fname = figs_path / f'{fname_stem}.png'
        f.savefig(fig_fname, bbox_inches='tight')

        # Create the tex for this figure
        tex_path = f'{tex_base_path}/figs/'
        tex_path = results_path / tex_path
        tex_path.mkdir(parents=True, exist_ok=True)

        tex_fname_path = tex_path/ f'{fname_stem}.tex'

        png_path = f'{rel_path}/{fname_stem}.png'

        create_latex_figure(tex_fname_path, png_path, caption=caption, short_caption=short_caption)


def write_latex_multi_column(dfx, hue_column = 'loss_function_', inner_column = 'include_T1_', \
                             table_name='table1', caption='Caption of the table', \
                             short_caption=None, show_df=True):

    df_all = None

    row_arrays_level_1 = []
    row_arrays_level_2 = []
    for group_name, df_group in dfx.groupby(hue_column):
        df_group = df_group.sort_values(inner_column)
        row_arrays_level_1+= list(df_group[hue_column])
        row_arrays_level_2+= list(df_group[inner_column])

        if isinstance(df_all, pd.DataFrame):
            df_all = pd.concat([df_all, df_group], axis=0, ignore_index=True)
        else:
            df_all = df_group    

    row_arrays = pd.MultiIndex.from_arrays([row_arrays_level_1,row_arrays_level_2])
    data = df_all[['mean dsc','median dsc','mean ppv','mean sensitivity', 'mean dscoeff']].to_numpy()

    dfx = pd.DataFrame(data, index=row_arrays, columns=['mean dsc','median dsc','mean ppv','mean sensitivity', 'mean dscoeff'])

    write_latex_table2D(dfx, table_name=table_name, caption=caption, short_caption=short_caption, multirow=True)

    if show_df:
        display(dfx)


def write_latex_multi_row(dfx, first_column = 'loss_function_', \
                          second_column = 'include_T1_', third_column=None, \
                          table_name='table1', caption='Caption of the table', \
                          short_caption=None, show_df=True):

    df_all = None

    row_arrays_level_1 = []
    row_arrays_level_2 = []
    row_arrays_level_3 = []
    dfx = dfx.sort_values(by=['mean dsc'], ascending=False)
    for group_name, df_group in dfx.groupby(first_column):

        if third_column is not None:
            df_group = df_group.sort_values(by=[second_column, third_column])
        else:
            df_group = df_group.sort_values(by=[second_column])

        row_arrays_level_1+= list(df_group[first_column])
        row_arrays_level_2+= list(df_group[second_column])
        if third_column is not None:
            row_arrays_level_3+= list(df_group[third_column])

        if isinstance(df_all, pd.DataFrame):
            df_all = pd.concat([df_all, df_group], axis=0, ignore_index=True)
        else:
            df_all = df_group    

    if third_column is not None:
        row_arrays = pd.MultiIndex.from_arrays([row_arrays_level_1,row_arrays_level_2,row_arrays_level_3])
    else:
        row_arrays = pd.MultiIndex.from_arrays([row_arrays_level_1,row_arrays_level_2])

    data = df_all[['mean dsc','median dsc','mean ppv','mean sensitivity']].to_numpy()

    dfx = pd.DataFrame(data, index=row_arrays, columns=['mean dsc','median dsc','mean ppv','mean sensitivity'])

    write_latex_table2D(dfx, table_name=table_name, caption=caption, short_caption=short_caption, multirow=True)

    if show_df:
        display(dfx)

def write_latex_table2D(df,
                        table_name='table1',
                        caption='Caption of the table',
                        short_caption=None,
                        column_format=None,
                        multirow=False,
                        multicolumn=False,
                        multicolumn_format=None,
                        index=True,
                        float_format='{:.4f}'.format):

    tables_path = results_path / tex_base_path / 'tables'
    tables_path.mkdir(parents=True, exist_ok=True)

    column_names_dict = {'mean dsc' : 'Mean DSC',
                         'median dsc' : 'Median DSC', 
                         'mean ppv' : 'Mean PPV',
                         'mean sensitivity': 'Mean Sensitivity',
                         'network_name_': 'Network Name',
                         'loss_function_': 'Loss Function',
                         'include_T1_': 'Images',
                         'augmentation_': 'Augmentations',
                         }

    df = df.rename(columns=column_names_dict)

    if short_caption is not None:
        caption_ = (caption, short_caption)
    else:
        caption_ = caption

    with open( tables_path / f'{table_name}.tex', 'w') as file: 
        file.write(df.to_latex(
            caption=caption_,
            escape=True,
            column_format=column_format,
            label=f'tab:{table_name}',
            float_format=float_format,                                             
            multirow=multirow,
            multicolumn=multicolumn,
            index=index,
            multicolumn_format=multicolumn_format
            )
        )


def write_to_input_commands_tex(tex_fname):

    FILELOCATION = str(tex_fname).replace(str(results_path), '.')
    input_str = r"""\input{FILELOCATION}""" +"\n"

    input_str = input_str.replace('FILELOCATION', FILELOCATION)

    input_commands = tex_fname.parent / 'input_commands.tex'
    print(input_str)

    f = open(input_commands, "a")
    f.write(input_str)
    f.close()   

    if '.tex' in str(tex_fname):
        f = open(results_path / tex_base_path / 'all_input_commands.tex', "a")
        f.write(input_str)
        f.close()   

    return 



def get_compare_labellers_df(patent_ids=[], multi_SN_RN_path=None, verbose=False):
    # data = compare_labellers_statistics_v3(patent_ids=patent_ids, multi_SN_RN_path=multi_SN_RN_path, verbose=verbose)

    # df = pd.DataFrame(data=data, columns=["patient_id", "TP", "FN", "FP", "DSC", "IOU"])
    # df = df.sort_values(['DSC'], ascending=False)
    # df = df.astype({'patient_id': 'int32'})

    def ppv(tp, fp):
        return (tp)/(tp+fp)

    def sensitivity(tp, fn):
        return (tp)/(tp+fn)

    def dscoeff(tp, fn, fp):
        return (2*tp)/(2*tp+fn+fp)

    df = get_compare_labellers_df_v3(patent_ids=patent_ids, multi_SN_RN_path=multi_SN_RN_path, verbose=verbose)
    df['PPV'] = df.apply(lambda x: ppv(x.TP, x.FP), axis=1)
    df['Sensitivity'] = df.apply(lambda x: sensitivity(x.TP, x.FN), axis=1)

    return df



def plot_from_kde_df(df, 
                     fname_stem=None, 
                     figsize=(16,6),
                     caption='', 
                     set_xlim=(0.5, 1),
                     short_caption=None):

    df = pd.melt(df, value_vars=['DSC', 'PPV', 'Sensitivity'], var_name='metric')
    sns.set(font_scale=2)

    f, ax = plt.subplots(1,1, figsize=figsize)
    # sns.histplot(data=df, hue='metric',x='value', stat='density', \
    #              kde=True, common_norm=False, bins=10, line_kws={'linewidth':3}, fill=False )#, kind="kde", fill=False, height=12, aspect=1.6, cut=0, bw_adjust=1)

    sns.kdeplot(data=df, hue='metric',x='value', \
                common_norm=False, linewidth=3, fill=False )#, kind="kde", fill=False, height=12, aspect=1.6, cut=0, bw_adjust=1)

    ax.set_xlabel('Metric value', fontsize=24)

    if set_xlim :
        ax.set_xbound(set_xlim)
        ax.set_xlim(set_xlim)        

    if fname_stem:

        rel_path = f'{tex_base_path}/images/'
        figs_path = results_path / rel_path
        figs_path.mkdir(parents=True, exist_ok=True)

        fig_fname = figs_path / f'{fname_stem}.png'
        f.savefig(fig_fname, bbox_inches='tight')

        # Create the tex for this figure
        tex_path = f'{tex_base_path}/figs/'
        tex_path = results_path / tex_path
        tex_path.mkdir(parents=True, exist_ok=True)

        tex_fname_path = tex_path/ f'{fname_stem}.tex'

        png_path = f'{rel_path}/{fname_stem}.png'

        create_latex_figure(tex_fname_path, png_path, caption=caption, short_caption=short_caption)




def create_3col_figure_by_model(patient_ids, 
                                model_name,
                                fname_stem = 'complete_model_results_9x9',
                                caption = '',
                                short_caption=None,
                                verbose=True):

    latex_str = r"""
\begin{figure}[!ht]
    \centering
    \begin{tabular}{cccccc}
    \rotatebox[origin=c]{90}{\textbf{RHEADING00}}  & ADJUSTBOX00 
    \rotatebox[origin=c]{90}{\textbf{RHEADING01}}  & ADJUSTBOX01 
    \rotatebox[origin=c]{90}{\textbf{RHEADING02}}  & ADJUSTBOX02 \\[-2mm]
    & SUBCAPTION00 & SUBCAPTION01 & SUBCAPTION02 \\[2mm]
    \rotatebox[origin=c]{90}{\textbf{RHEADING10}}  & ADJUSTBOX10 
    \rotatebox[origin=c]{90}{\textbf{RHEADING11}}  & ADJUSTBOX11 
    \rotatebox[origin=c]{90}{\textbf{RHEADING12}}  & ADJUSTBOX12 \\[-2mm]
    & SUBCAPTION10 & SUBCAPTION11 & SUBCAPTION12 \\[2mm]
    \rotatebox[origin=c]{90}{\textbf{RHEADING20}}  & ADJUSTBOX20 
    \rotatebox[origin=c]{90}{\textbf{RHEADING21}}  & ADJUSTBOX21 
    \rotatebox[origin=c]{90}{\textbf{RHEADING22}}  & ADJUSTBOX22 \\[-2mm]
    & SUBCAPTION20 & SUBCAPTION21 & SUBCAPTION22 \\[2mm]
    \end{tabular}
    \caption[SHORTCAPTION]{FULLCAPTION}
    \label{fig:LABEL}
\end{figure}
"""

    experiment_name = model_name.replace('.pt', '').replace('-', '_')

    # NO NEED FOR THIS HEADER ROW?
    # & \textbf{CHEADING0} & \textbf{CHEADING1} & \textbf{CHEADING2} \\[2mm]   
    #CHEADINGS = ['Baseline', 'T2 Only', 'T1 and T2'], 
    #replace_dict = {'CHEADING0' : CHEADINGS[0], 'CHEADING1' : CHEADINGS[1], 'CHEADING2' : CHEADINGS[2], 'FULLCAPTION' : CAPTION, 'LABEL' : fig_fname.stem}
    
    figs_path = results_path / f"{tex_base_path}/figs"
    figs_path.mkdir(parents=True, exist_ok=True)
    fig_fname = figs_path/ f'{fname_stem}.tex'

    ADJUSTBOX_ = lambda x: r"""\adjustbox{valign=m,vspace=1pt}{\includegraphics[width=.29\linewidth]{FILENAME}}""".replace('FILENAME', x)

    replace_dict = {'FULLCAPTION' : caption, 'LABEL' : fig_fname.stem}

    if short_caption:
        short_caption_ = f'[{short_caption}]'
    else:
        short_caption_ = ''

    replace_dict.update({'[SHORTCAPTION]' : short_caption_})

    p_idx = 0
    for row_idx in range(3):
        for col_idx in range(3):

            if p_idx < len(patient_ids):
                patient_id = patient_ids[p_idx]

                latex_path = f"{tex_base_path}/images/{str(patient_id)}"
            
                # if Path(results_path / f'{latex_path}/{experiment_name}.png').is_file() and  \
                # Path(results_path / f'{latex_path}/{experiment_name}.txt').is_file():

                replace_dict[f'ADJUSTBOX{row_idx}{col_idx}'] =  ADJUSTBOX_(f'{latex_path}/{experiment_name}.png')
                replace_dict[f'SUBCAPTION{row_idx}{col_idx}'] =  get_caption(results_path / f'{latex_path}/{experiment_name}.txt')
                replace_dict[f'RHEADING{row_idx}{col_idx}'] =  f'Subject: {str(patient_id)}'

            else:
                replace_dict[f'ADJUSTBOX{row_idx}{col_idx}'] =  ''
                replace_dict[f'SUBCAPTION{row_idx}{col_idx}'] =  ''
                replace_dict[f'RHEADING{row_idx}{col_idx}'] = ''

            p_idx+=1

    for key, value in replace_dict.items():
        latex_str = latex_str.replace(key, value)

    f = open(fig_fname, "w")
    f.write(latex_str)
    f.close()   

    write_to_input_commands_tex(fig_fname)

    if verbose: print(latex_str)

def model_idx_to_name(df, df_idx):
    return df.loc[df_idx,:]['model_name']

def get_pid_ordered_by_dsc(model_name_):
    images_path = results_path / tex_base_path / 'images'

    pid2dsc = {}
    for root, dirs, files in os.walk(images_path):
        root_path = Path(root)
        for file in files:
            if file == model_name_.replace('-', '_').replace('.pt', '.txt'):
                caption_ = get_caption(root_path / file)
                regex = re.search(r'DSC=(.*) (.*)', caption_)
                if regex:
                    DSC_ = float(regex.group(1))
                    patient_id = int(root_path.stem)
                    pid2dsc[patient_id] = DSC_

    return list(dict(sorted(pid2dsc.items(), key=lambda item: item[1])).keys())

def plot_density_plot_by_df_idx(df, df_box, df_idx, \
                                fname_stem='density_plot_by_model', delta=0.1, \
                                figsize=(16,6)):
    
    model_name_ = df.loc[df_idx,:]['model_name']
    network_name_ = df.loc[df_idx,:]['network_name_']
    include_T1_ = df.loc[df_idx,:]['include_T1_']
    augmentation_ = df.loc[df_idx,:]['augmentation_']
    loss_function_ = df.loc[df_idx,:]['loss_function_']

    # First create plots for all samples
    df_box_t = df_box[df_box['model_name'] == model_name_]    
    num_samples_all = df_box_t.shape[0]
    
    column_names_dict = {'ppv': 'PPV', 'sensitivity' : 'Sensitivity', 'dsc' : 'DSC'}
    df_box_t = df_box_t.rename(columns=column_names_dict)

    caption = f'Density plot of DSC, PPV and Sensitivity on {network_name_} model ({loss_function_}, {augmentation_}, {include_T1_})'
    print(caption)

    plot_from_kde_df(df=df_box_t, \
                     figsize=figsize,
                     fname_stem=f'{fname_stem}_all',
                     caption=caption)


    # Next create samples for samples above threshold
    df_box_t0 = df_box_t[abs(df_box_t['PPV'] - df_box_t['Sensitivity']) <= delta]
    num_samples_0 = df_box_t0.shape[0]
    
    caption = f'Density plot of DSC, PPV and Sensitivity on {network_name_} model ' + \
              f'({loss_function_}, {augmentation_}, {include_T1_}). ' + \
              f'Figure only shows samples in the validation set where PPV > Sensitivity by at least {delta} ({num_samples_0} samples) '
    
    print(caption)

    plot_from_kde_df(df=df_box_t0, \
                     figsize=figsize,
                     fname_stem=f'{fname_stem}_1',
                     caption=caption)
    
    # Next create samples for samples above threshold
    df_box_t1 = df_box_t[df_box_t['PPV'] > df_box_t['Sensitivity']+delta]
    num_samples_1 = df_box_t1.shape[0]
    
    caption = f'Density plot of DSC, PPV and Sensitivity on {network_name_} model ' + \
              f'({loss_function_}, {augmentation_}, {include_T1_}). ' + \
              f'Figure only shows samples in the validation set where PPV > Sensitivity by at least {delta} ({num_samples_1} samples) '
    
    print(caption)

    plot_from_kde_df(df=df_box_t1, \
                     figsize=figsize,
                     fname_stem=f'{fname_stem}_2',
                     caption=caption)

    
    # Next create samples for samples above threshold
    df_box_t2 = df_box_t[df_box_t['PPV'] < df_box_t['Sensitivity']-delta]
    num_samples_2 = df_box_t2.shape[0]
    
    caption = f'Density plot of DSC, PPV and Sensitivity on {network_name_} model ' + \
              f'({loss_function_}, {augmentation_}, {include_T1_}). ' + \
              f'Figure only shows samples in the validation set where PPV < Sensitivity by at least {delta} ({num_samples_2} samples) '
    
    print(caption)

    plot_from_kde_df(df=df_box_t2, \
                     figsize=figsize,
                     fname_stem=f'{fname_stem}_3',
                     caption=caption)
