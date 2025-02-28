"""
Experimental Script Generator    Script  ver： Nov 29th 13:00

for linux servers

todo fix train and test alternatively
"""
import argparse
import os.path


def zero_trans_mystrlr_to_float(in_str):
    # EG: '305' -> 0.0005
    front = '0.'
    num_of_zero = int(in_str[0])
    end = in_str[-1]
    for i in range(num_of_zero):
        front = front + '0'
    front = front + end

    out_float = float(front)

    return out_float


def zero_trans_floatlr_to_mystrlr(in_float):
    # EG: 0.0005 -> '305'
    in_string = "%.20f" % in_float
    zero_counts = 0

    for i in range(len(in_string) - 2):
        # print(string[i+2])
        if in_string[i + 2] == '0':
            zero_counts += 1
        else:
            cut = i
            break

    trans_output = str(zero_counts) + '0' + in_string[(cut + 2):]

    last_zeros = 0
    for i in trans_output[::-1]:
        if i == '0':
            last_zeros += 1
        else:
            break
    trans_output = trans_output[0:0 - last_zeros]

    return trans_output


def remove_nohup_ignoring_input_at_first_line(directory='./'):
    """
    read the .sh files at the directory, remove the first line if it's 'nohup: ignoring input\n'
    """
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".sh"):
                file_path = os.path.join(root, file_name)

                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    # print(lines)

                modified_lines = [line for line in lines if line != "nohup: ignoring input\n"]
                with open(file_path, 'w') as file:
                    file.writelines(modified_lines)

                print('file_path:', file_path, 'has been cleaned')


def concatenate_the_lines_from_several_files(directory='./', cat_file='0.sh'):
    cat_file_path = os.path.join(directory, cat_file)
    all_lines = ["#!/bin/sh\n", ]

    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".sh"):
                file_path = os.path.join(root, file_name)

                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    # print(lines)

                modified_lines = [line for line in lines if line != "#!/bin/sh\n"]
                all_lines.extend(modified_lines)
                print('file_path:', file_path, 'has taken')

    with open(cat_file_path, 'w') as file:
        file.writelines(all_lines)


def print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_Token_num=20,
                                           Prompt_input=False):
    Pre_Trained_model_path = os.path.join(Pre_Trained_model_path_PATH, model_weight_name)
    VPT_backbone_model_path = os.path.join(Pre_Trained_model_path_PATH, 'ViT_b16_224_Imagenet.pth')
    if not Prompt_input:
        # send a ViT model inside and then do the ViT + finetuning;
        # In VPT versions: we build VPT backbone with the ViT weight, then do finetuning and prompting
        # ViT + finetuning
        print(
            'python Train.py --gpu_idx ' + GPU_idx + ' --edge_size 224 --data_augmentation_mode ' + data_augmentation_mode
            + ' --lr ' + lr + ' --lrf ' + lrf + ' --enable_tensorboard --model_idx ViT_base_' + model_weight_idx + '_'
            + lr_mystr + '_lf' + lrf_mystr + '_finetuning_' + dataset_name + '_CLS --dataroot ' + str(dataroot)
            + ' --draw_root ' + draw_root + ' --Pre_Trained_model_path ' + Pre_Trained_model_path
            + ' --model_path ' + save_model_PATH)
        print(
            'python Test.py --gpu_idx ' + GPU_idx + ' --edge_size 224 --data_augmentation_mode ' + data_augmentation_mode
            + ' --model_idx ViT_base_' + model_weight_idx + '_' + lr_mystr + '_lf' + lrf_mystr + '_finetuning_'
            + dataset_name + '_CLS --dataroot ' + str(dataroot) + ' --draw_root ' + draw_root + ' --model_path '
            + save_model_PATH)
        # VPT + prompting
        print(
            'python Train.py --gpu_idx ' + GPU_idx + ' --edge_size 224 --data_augmentation_mode ' + data_augmentation_mode
            + ' --lr ' + lr + ' --lrf ' + lrf + ' --enable_tensorboard --model_idx ViT_base_' + model_weight_idx
            + '_PromptDeep_' + str(Prompt_Token_num) + '_' + lr_mystr + '_lf' + lrf_mystr + '_prompting_' + dataset_name
            + '_CLS --PromptTuning Deep --Prompt_Token_num ' + str(Prompt_Token_num) + ' --dataroot ' + str(
                dataroot) + ' --draw_root ' + draw_root
            + ' --Pre_Trained_model_path ' + Pre_Trained_model_path + ' --model_path ' + save_model_PATH)
        print(
            'python Test.py --gpu_idx ' + GPU_idx + ' --edge_size 224 --data_augmentation_mode ' + data_augmentation_mode
            + ' --model_idx ViT_base_' + model_weight_idx + '_PromptDeep_' + str(
                Prompt_Token_num) + '_' + lr_mystr + '_lf' + lrf_mystr
            + '_prompting_' + dataset_name + '_CLS --PromptTuning Deep --Prompt_Token_num ' + str(
                Prompt_Token_num) + ' --dataroot ' + str(dataroot) + ' --draw_root '
            + draw_root + ' --Pre_Trained_model_path ' + Pre_Trained_model_path + ' --model_path ' + save_model_PATH)
        # VPT + finetuning
        print(
            'python Train.py --gpu_idx ' + GPU_idx + ' --edge_size 224 --data_augmentation_mode ' + data_augmentation_mode
            + ' --lr ' + lr + ' --lrf ' + lrf + ' --enable_tensorboard --model_idx ViT_base_' + model_weight_idx
            + '_PromptDeep_' + str(
                Prompt_Token_num) + '_' + lr_mystr + '_lf' + lrf_mystr + '_finetuning_' + dataset_name
            + '_CLS --PromptTuning Deep --Prompt_Token_num ' + str(
                Prompt_Token_num) + ' --PromptUnFreeze --dataroot ' + str(dataroot) + ' --draw_root ' + draw_root
            + ' --Pre_Trained_model_path ' + Pre_Trained_model_path + ' --model_path ' + save_model_PATH)
        print(
            'python Test.py --gpu_idx ' + GPU_idx + ' --edge_size 224 --data_augmentation_mode ' + data_augmentation_mode
            + ' --model_idx ViT_base_' + model_weight_idx + '_PromptDeep_' + str(
                Prompt_Token_num) + '_' + lr_mystr + '_lf' + lrf_mystr
            + '_finetuning_' + dataset_name + '_CLS --PromptTuning Deep --Prompt_Token_num ' + str(
                Prompt_Token_num) + ' --PromptUnFreeze --dataroot ' + str(dataroot)
            + ' --draw_root ' + draw_root + ' --model_path ' + save_model_PATH)
    else:
        # send a VPT prompt state inside to build the prompt tokens
        # we build VPT backbone with the ViT-timm weight, then do finetuning and prompting
        # fixme notice here Pre_Trained_model_path is actually the trained prompt state path
        # VPT + prompting
        print(
            'python Train.py --gpu_idx ' + GPU_idx + ' --edge_size 224 --data_augmentation_mode ' + data_augmentation_mode
            + ' --lr ' + lr + ' --lrf ' + lrf + ' --enable_tensorboard --model_idx ViT_base_' + model_weight_idx
            + '_PromptDeep_' + str(Prompt_Token_num) + '_' + lr_mystr + '_lf' + lrf_mystr + '_prompting_' + dataset_name
            + '_CLS --PromptTuning Deep --Prompt_Token_num ' + str(Prompt_Token_num) + ' --dataroot ' + str(
                dataroot) + ' --draw_root ' + draw_root
            + ' --Pre_Trained_model_path ' + VPT_backbone_model_path + ' --Prompt_state_path ' + Pre_Trained_model_path + ' --model_path ' + save_model_PATH)
        print(
            'python Test.py --gpu_idx ' + GPU_idx + ' --edge_size 224 --data_augmentation_mode ' + data_augmentation_mode
            + ' --model_idx ViT_base_' + model_weight_idx + '_PromptDeep_' + str(
                Prompt_Token_num) + '_' + lr_mystr + '_lf' + lrf_mystr
            + '_prompting_' + dataset_name + '_CLS --PromptTuning Deep --Prompt_Token_num ' + str(
                Prompt_Token_num) + ' --dataroot ' + str(dataroot) + ' --draw_root '
            + draw_root + ' --Pre_Trained_model_path ' + VPT_backbone_model_path + ' --model_path ' + save_model_PATH)
        # VPT + finetuning
        print(
            'python Train.py --gpu_idx ' + GPU_idx + ' --edge_size 224 --data_augmentation_mode ' + data_augmentation_mode
            + ' --lr ' + lr + ' --lrf ' + lrf + ' --enable_tensorboard --model_idx ViT_base_' + model_weight_idx
            + '_PromptDeep_' + str(
                Prompt_Token_num) + '_' + lr_mystr + '_lf' + lrf_mystr + '_finetuning_' + dataset_name
            + '_CLS --PromptTuning Deep --Prompt_Token_num ' + str(
                Prompt_Token_num) + ' --PromptUnFreeze --dataroot ' + str(dataroot) + ' --draw_root ' + draw_root
            + ' --Pre_Trained_model_path ' + VPT_backbone_model_path + ' --Prompt_state_path ' + Pre_Trained_model_path + ' --model_path ' + save_model_PATH)
        print(
            'python Test.py --gpu_idx ' + GPU_idx + ' --edge_size 224 --data_augmentation_mode ' + data_augmentation_mode
            + ' --model_idx ViT_base_' + model_weight_idx + '_PromptDeep_' + str(
                Prompt_Token_num) + '_' + lr_mystr + '_lf' + lrf_mystr
            + '_finetuning_' + dataset_name + '_CLS --PromptTuning Deep --Prompt_Token_num ' + str(
                Prompt_Token_num) + ' --PromptUnFreeze --dataroot ' + str(dataroot)
            + ' --draw_root ' + draw_root + ' --model_path ' + save_model_PATH)

    print('')


def write_PuzzleTuning_comparison_script(lr_mystr, lrf_mystr, data_augmentation_mode, dataset_name, GPU_idx='0'):
    """
    In PuzzleTuning comparison experiments we put
    datasets at: --dataroot /root/autodl-tmp/datasets
    Pre_Trained_model_path /root/autodl-tmp/pre_trained_models  # output_models (not applicable for comparison)
    Prompt_state_path (not applicable for comparison) /root/autodl-tmp/output_models
    save the training model at: model_path /root/autodl-tmp/saved_models
    draw_root /root/autodl-tmp/PuzzleTuning_Comparison/[*lr*_*lrf*_*dataset_name*]

    """
    dataroot_PATH = '/root/autodl-tmp/datasets'
    Pre_Trained_model_path_PATH = '/root/autodl-tmp/pre_trained_models'
    save_model_PATH = '/root/autodl-tmp/saved_models'
    draw_root_PATH = '/root/autodl-tmp/PuzzleTuning_Comparison'

    data_augmentation_mode = str(data_augmentation_mode)
    GPU_idx = str(GPU_idx)

    lr = str(zero_trans_mystrlr_to_float(lr_mystr))
    lrf = '0.' + str(lrf_mystr)

    experiment_idx = lr_mystr + '_lf' + lrf_mystr + '_' + dataset_name

    dataroot = os.path.join(dataroot_PATH, dataset_name + '_CLS')
    draw_root = os.path.join(draw_root_PATH, experiment_idx)

    # PuzzleTuning official version:
    # we pre-trained VPT prompt tokens, and use the timm ViT as backbone
    print('#SAE-timm-start_promptstate')  # SAE+VPT start with timm
    model_weight_idx = 'ViT_base_timm_PuzzleTuning_SAE_E_199_promptstate'
    model_weight_name = 'ViT_b16_224_timm_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=True)

    # Comparison methods:

    # For the comparison methods: we trained ViT, so we use ViT + ft first,
    # and then, put it as vpt 's backbone in prompting and VPT finetuning.
    print('#空白对比')
    model_weight_idx = 'random'
    model_weight_name = 'ViT_b16_224_Random_Init.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    print('#timm对比')
    model_weight_idx = 'timm'
    model_weight_name = 'ViT_b16_224_Imagenet.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    print('#MAEImageNet对比')
    model_weight_idx = 'MAEImageNet'
    model_weight_name = 'ViT_b16_224_MAEImageNet_Init.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    print('#mae对比')
    model_weight_idx = 'timm_mae_CPIAm_E100'
    model_weight_name = 'ViT_b16_224_timm_mae_ALL_100.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    print('#moco对比')
    model_weight_idx = 'timm_moco_CPIAm_E100'
    model_weight_name = 'ViT_b16_224_timm_moco_ALL_100.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    print('#dino对比')
    model_weight_idx = 'timm_dino_CPIAm_E100'
    model_weight_name = 'ViT_b16_224_timm_dino_ALL_100.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    print('#BYOL对比')
    model_weight_idx = 'timm_BYOL_CPIAm_E50'
    model_weight_name = 'ViT_b16_224_timm_BYOL_ALL_50.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    print('#GCMAE对比')
    model_weight_idx = 'timm_GCMAE_CPIAm_E80'
    model_weight_name = 'ViT_b16_224_timm_GCMAE_ALL_80.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    print('#SDMAE对比')
    model_weight_idx = 'timm_SDMAE_CPIAm_E80'
    model_weight_name = 'ViT_b16_224_timm_SDMAE_ALL_80.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    print('#SIMMIM对比')
    model_weight_idx = 'timm_SIMMIM_CPIAm_E200'
    model_weight_name = 'ViT_b16_224_timm_SIMMIM_ALL_200.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    print('#SIMCLR对比')
    model_weight_idx = 'timm_SIMCLR_CPIAm_E100'
    model_weight_name = 'ViT_b16_224_timm_SIMCLR_ALL_100.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    # Ablation versions:

    # For ablation SAE-ViT version, we pre-trained ViT, so we use ViT + ft first,
    # and then, put it as vpt 's backbone in prompting and VPT finetuning.
    print('#PuzzleTuning_SAE_ViT-CPIA对比')
    model_weight_idx = 'timm_PuzzleTuning_SAE_E_199'
    model_weight_name = 'ViT_b16_224_timm_PuzzleTuning_SAE_CPIAm_E_199.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    print('#SAE_fixp16fixr25-timm-start')  # SAE_fixp16fixr25+ViT start with timm
    model_weight_idx = 'ViT_base_timm_PuzzleTuning_SAE_fixp16fixr25_E_199'
    model_weight_name = 'ViT_b16_224_timm_PuzzleTuning_SAE_fixp16fixr25_CPIAm_E_199.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    print('#SAE_fixp16ratiodecay-timm-start')  # SAE_fixp16ratiodecay+ViT start with timm
    model_weight_idx = 'ViT_base_timm_PuzzleTuning_SAE_fixp16ratiodecay_E_199'
    model_weight_name = 'ViT_b16_224_timm_PuzzleTuning_SAE_fixp16ratiodecay_CPIAm_E_199.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=False)

    # For ablation SAE-VPT version, we pre-trained VPT prompt tokens, and use the timm ViT as backbone
    print('#MAE-VPT_promptstate')  # MAE+VPT
    model_weight_idx = 'timm_mae_Prompt_CPIAm_E199_promptstate'
    model_weight_name = 'ViT_b16_224_timm_PuzzleTuning_MAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=True)

    print('#SAE-MAE-start_promptstate')  # SAE+VPT start with MAEImageNet
    model_weight_idx = 'ViT_base_MAEImageNet_PuzzleTuning_SAE_E_199_promptstate'
    model_weight_name = 'ViT_b16_224_MAEImageNet_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=True)

    print('#SAE-Random-start_promptstate')  # SAE+VPT start with Random
    model_weight_idx = 'ViT_base_Random_PuzzleTuning_SAE_E_199_promptstate'
    model_weight_name = 'ViT_b16_224_Random_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=True)

    print('#SAE_fixp16fixr25-timm-start_promptstate')  # SAE_fixp16fixr25+VPT start with timm
    model_weight_idx = 'ViT_base_timm_PuzzleTuning_SAE_fixp16fixr25_E_199_promptstate'
    model_weight_name = 'ViT_b16_224_timm_PuzzleTuning_SAE_fixp16fixr25_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=True)

    print('#SAE_fixp16ratiodecay-timm-start_promptstate')  # SAE_fixp16ratiodecay+VPT start with timm
    model_weight_idx = 'ViT_base_timm_PuzzleTuning_SAE_fixp16ratiodecay_E_199_promptstate'
    model_weight_name = 'ViT_b16_224_timm_PuzzleTuning_SAE_fixp16ratiodecay_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth'
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=True)

    print('')
    print('cd /home/pancreatic-cancer-diagnosis-tansformer/code/utils')
    record_dir = os.path.join(draw_root, 'CSV_logs')
    print('python check_log_json.py --enable_notify --draw_root ' + draw_root + ' --record_dir ' + record_dir)
    print('cd /home/pancreatic-cancer-diagnosis-tansformer/code')


def write_additional_PuzzleTuning_comparison_script(add_idx, lr_mystr, lrf_mystr, data_augmentation_mode, dataset_name,
                                                    model_weight_idx='timm_mae_CPIAm_E100',
                                                    model_weight_name='ViT_b16_224_timm_mae_ALL_100.pth',
                                                    GPU_idx='0', Prompt_input=False):
    """
    In PuzzleTuning comparison experiments we put
    datasets at: --dataroot /root/autodl-tmp/datasets
    Pre_Trained_model_path /root/autodl-tmp/pre_trained_models  # output_models (not applicable for comparison)
    Prompt_state_path (not applicable for comparison) /root/autodl-tmp/output_models
    save the training model at: model_path /root/autodl-tmp/saved_models
    draw_root /root/autodl-tmp/PuzzleTuning_Comparison/[*lr*_*lrf*_*dataset_name*]

    # fixme the additional experiments settings need to manually set!!!
    in the additional experiments, we save the runs to
    draw_root /root/autodl-tmp/runs/[*lr*_*lrf*_*dataset_name*]
    and then copy a duplicates to /root/autodl-tmp/PuzzleTuning_Comparison/[*lr*_*lrf*_*dataset_name*]

    """
    dataroot_PATH = '/root/autodl-tmp/datasets'
    Pre_Trained_model_path_PATH = '/root/autodl-tmp/pre_trained_models'
    save_model_PATH = '/root/autodl-tmp/saved_models'
    draw_root_PATH = '/root/autodl-tmp/runs'
    copy_to_draw_root_PATH = '/root/autodl-tmp/PuzzleTuning_Comparison'

    data_augmentation_mode = str(data_augmentation_mode)
    GPU_idx = str(GPU_idx)

    lr = str(zero_trans_mystrlr_to_float(lr_mystr))
    lrf = '0.' + str(lrf_mystr)

    experiment_idx = lr_mystr + '_lf' + lrf_mystr + '_' + dataset_name
    add_experiment_idx = add_idx + '_' + lr_mystr + '_lf' + lrf_mystr + '_' + dataset_name

    dataroot = os.path.join(dataroot_PATH, dataset_name + '_CLS')
    # additional exp runs path
    draw_root = os.path.join(draw_root_PATH, add_experiment_idx)
    # basic all exp runs path
    copy_draw_root = os.path.join(copy_to_draw_root_PATH, experiment_idx)

    print('# Additional ' + add_idx)
    print_a_PuzzleTuning_comparison_script(model_weight_idx, model_weight_name, lr, lrf, lr_mystr, lrf_mystr,
                                           dataset_name, dataroot, draw_root, Pre_Trained_model_path_PATH,
                                           save_model_PATH, data_augmentation_mode, GPU_idx, Prompt_input=Prompt_input)
    print('')
    print('cd /home/pancreatic-cancer-diagnosis-tansformer/code/utils')
    # update the total record
    print('')
    print('cp -r ' + draw_root + '/*' + ' ' + copy_draw_root)
    record_dir = os.path.join(copy_draw_root, 'CSV_logs')
    print('python check_log_json.py --draw_root ' + copy_draw_root + ' --record_dir ' + record_dir)

    # update the additional runs and send to notify
    record_dir = os.path.join(draw_root, add_experiment_idx)
    print('python check_log_json.py --enable_notify --draw_root ' + draw_root + ' --record_dir ' + record_dir)

    print('cd /home/pancreatic-cancer-diagnosis-tansformer/code')


def write_CLS_script(model_idxs, data_augmentation_mode, edge_size, batch_size, lr, lrf, enable_tensorboard,
                     test_enable_attention_check, dataset_name, dataroot, model_path, draw_root, augmentation_name=None,
                     test_aug_attention_check=False, offline_augmentation=False):

    data_augmentation_mode = str(data_augmentation_mode)
    batch_size = str(batch_size)
    lr_name = zero_trans_floatlr_to_mystrlr(lr) if type(lr) == float else lr
    lr = str(lr) if type(lr) == float else str(zero_trans_mystrlr_to_float(lr))
    lf_name = str(int(100 * lrf)) if type(lrf) == float else lrf
    lrf = str(float('0.'+lrf)) if type(lrf) != float else str(lrf)
    dataroot = dataroot + dataset_name + '_CLS'

    # Train
    for model_idx in model_idxs:

        # alter the edge size for certain models
        if model_idx in ['cross_former', 'convit', 'visformer', 'ViT_h']:
            edge_size = '224'
        else:
            edge_size = str(edge_size)

        head = 'python Train.py --model_idx ' + model_idx + '_' + edge_size + '_' + lr_name + '_PT_lf' \
               + lf_name + '_b' + batch_size + '_'
        aug_data = dataset_name
        mid = '_CLS '
        tail = '--edge_size ' + edge_size + ' --data_augmentation_mode ' + data_augmentation_mode + ' --batch_size ' + \
               batch_size + ' --lr ' + lr + ' --lrf ' + lrf + ' --dataroot ' + dataroot + ' --model_path ' \
               + model_path + ' --draw_root ' + draw_root

        if enable_tensorboard is True:
            mid = mid + '--enable_tensorboard '
        if augmentation_name is not None:
            aug_data = aug_data + '_' + str(augmentation_name)
            mid = mid + '--augmentation_name ' + str(augmentation_name) + ' '
        if offline_augmentation:
            mid = mid + '--confusing_training '

        print(head + aug_data + mid + tail)
        print('')

    # Test
    for model_idx in model_idxs:

        # alter the edge size for certain models
        if model_idx in ['cross_former', 'convit', 'visformer', 'ViT_h']:
            edge_size = '224'
        else:
            edge_size = str(edge_size)

        head = 'python Test.py --model_idx ' + model_idx + '_' + edge_size + '_' + lr_name + '_PT_lf' + lf_name + '_b' \
               + batch_size + '_'
        aug_data = dataset_name
        mid = '_CLS '
        tail = '--edge_size ' + edge_size + ' --data_augmentation_mode ' + data_augmentation_mode + ' --dataroot ' \
               + dataroot + ' --model_path ' + model_path

        if augmentation_name is not None:
            aug_data = aug_data + '_' + str(augmentation_name)

        if test_enable_attention_check is True:
            mid = mid + '--enable_attention_check '

        print(head + aug_data + mid + tail + ' --draw_root ' + draw_root)
        print('')

        if test_aug_attention_check is True and augmentation_name is not None:
            mid = mid + '--data_augmentation_mode ' + data_augmentation_mode + ' '
            # another test for imaging
            print(head + aug_data + mid + tail + ' --draw_root ' + os.path.join(draw_root, 'imaging_results'))
            print('')


def write_CLS_AUG_script(model_idx, augmentation_names, data_augmentation_mode, edge_size, batch_size, lr, lrf, enable_tensorboard,
                         test_enable_attention_check, test_aug_attention_check, dataset_name, dataroot, model_path,
                         draw_root,offline_augmentation=False):
    # one model with multiple augmentation_names
    model_idxs = [model_idx,]

    for augmentation_name in augmentation_names:
        write_CLS_script(model_idxs, data_augmentation_mode, edge_size, batch_size, lr, lrf, enable_tensorboard,
                     test_enable_attention_check, dataset_name, dataroot, model_path, draw_root, augmentation_name,
                     test_aug_attention_check, offline_augmentation)


def write_MIL_script(model_idxs, data_augmentation_mode, edge_size, batch_size, patch_size, lr, lrf, enable_tensorboard,
                     test_enable_attention_check, dataset_name, dataroot, model_path, draw_root, imaging_root=None):
    # imaging_root 是放画图的检查的路径，可以和draw一样
    if imaging_root == None:
        imaging_root = draw_root

    data_augmentation_mode = str(data_augmentation_mode)
    edge_size = str(edge_size)
    batch_size = str(batch_size)
    patch_size = str(patch_size)
    lr_name = zero_trans_floatlr_to_mystrlr(lr)
    lr = str(lr)
    lf_name = str(int(100 * lrf))
    lrf = str(lrf)
    dataroot = dataroot + dataset_name + '_MIL'
    CLS_dataroot = dataroot + dataset_name + '_CLS'

    for model_idx in model_idxs:
        if enable_tensorboard is True:
            print('python MIL_train.py --model_idx ' + model_idx + '_' + edge_size + '_' + lr_name
                  + '_PT_lf' + lf_name + '_b' + batch_size + '_p' + patch_size + '_' + dataset_name +
                  '_MIL --edge_size ' + edge_size + ' --data_augmentation_mode ' + data_augmentation_mode +
                  ' --batch_size ' + batch_size + ' --patch_size ' + patch_size + ' --lr ' + lr + ' --lrf '
                  + lrf + ' --enable_tensorboard --dataroot ' + dataroot + ' --model_path ' + model_path
                  + ' --draw_root ' + draw_root)
            print('')
        else:
            print('python MIL_train.py --model_idx ' + model_idx + '_' + edge_size + '_' + lr_name
                  + '_PT_lf' + lf_name + '_b' + batch_size + '_p' + patch_size + '_' + dataset_name +
                  '_MIL --edge_size ' + edge_size + ' --data_augmentation_mode ' + data_augmentation_mode +
                  ' --batch_size ' + batch_size + ' --patch_size ' + patch_size + ' --lr ' + lr + ' --lrf '
                  + lrf + ' --dataroot ' + dataroot + ' --model_path ' + model_path + ' --draw_root ' + draw_root)
            print('')

    for model_idx in model_idxs:
        print('python MIL_test.py --model_idx ' + model_idx + '_' + edge_size + '_' + lr_name
              + '_PT_lf' + lf_name + '_b' + batch_size + '_p' + patch_size + '_' + dataset_name +
              '_MIL --edge_size ' + edge_size + ' --patch_size ' + patch_size +
              ' --batch_size 1 --data_augmentation_mode ' + data_augmentation_mode + ' --dataroot ' +
              dataroot + ' --model_path ' + model_path + ' --draw_root ' + draw_root)
        print('')

        if test_enable_attention_check is True:  # 设置多个batch的实验
            print('python Test.py --model_idx ' + model_idx + '_' + edge_size + '_' + lr_name
                  + '_PT_lf' + lf_name + '_b' + batch_size + '_p' + patch_size + '_' + dataset_name +
                  '_MIL --edge_size ' + edge_size + ' --data_augmentation_mode ' + data_augmentation_mode +
                  ' --MIL_Stripe --enable_attention_check --check_minibatch 10' +
                  ' --dataroot ' + CLS_dataroot + ' --model_path ' + model_path +
                  ' --draw_root ' + imaging_root)
            print('')
            print('python MIL_test.py --model_idx ' + model_idx + '_' + edge_size + '_' + lr_name
                  + '_PT_lf' + lf_name + '_b' + batch_size + '_p' + patch_size + '_' + dataset_name +
                  '_MIL --shuffle_attention_check --MIL_Stripe --edge_size ' + edge_size +
                  ' --data_augmentation_mode ' + data_augmentation_mode +
                  ' --shuffle_dataloader --batch_size 4 --check_minibatch 10' + ' --patch_size ' + patch_size +
                  ' --dataroot ' + dataroot + ' --model_path ' + model_path +
                  ' --draw_root ' + imaging_root)
            print('')
            print('python MIL_test.py --model_idx ' + model_idx + '_' + edge_size + '_' + lr_name
                  + '_PT_lf' + lf_name + '_b' + batch_size + '_p' + patch_size + '_' + dataset_name +
                  '_MIL --shuffle_attention_check --MIL_Stripe --edge_size ' + edge_size +
                  ' --data_augmentation_mode ' + data_augmentation_mode +
                  ' --batch_size 4 --check_minibatch 10' + ' --patch_size ' + patch_size +
                  ' --dataroot ' + dataroot + ' --model_path ' + model_path +
                  ' --draw_root ' + imaging_root)
            print('')
            print('python MIL_test.py --model_idx ' + model_idx + '_' + edge_size + '_' + lr_name
                  + '_PT_lf' + lf_name + '_b' + batch_size + '_p' + patch_size + '_' + dataset_name +
                  '_MIL --shuffle_attention_check --MIL_Stripe --edge_size ' + edge_size +
                  ' --data_augmentation_mode ' + data_augmentation_mode +
                  ' --batch_size 1 --check_minibatch 10' + ' --patch_size ' + patch_size +
                  ' --dataroot ' + dataroot + ' --model_path ' + model_path +
                  ' --draw_root ' + imaging_root)
            print('')

        else:
            print('python Test.py --model_idx ' + model_idx + '_' + edge_size + '_' + lr_name
                  + '_PT_lf' + lf_name + '_b' + batch_size + '_p' + patch_size + '_' + dataset_name +
                  '_MIL --edge_size ' + edge_size + ' --data_augmentation_mode ' + data_augmentation_mode +
                  ' --MIL_Stripe --dataroot ' + CLS_dataroot + ' --model_path ' + model_path +
                  ' --draw_root ' + draw_root)
            print('')


'''
if __name__ == '__main__':

    print('#!/bin/sh')
    print('')
    # CLS-MIL调参的第一步是使用一个经验参数进行简单摸索，看看大家结果大概是多少，同时和文献进行对比
    # 首先摸索CLS对比实验结果
    model_idxs = ['ViT', 'vgg16', 'vgg19', 'mobilenetv3', 'inceptionv3', 'xception',
                  'ResNet50', 'efficientnet_b3', 'swin_b', 'ResN50_ViT', 'conformer', 'cross_former']

    batch_size = 8
    dataset_name = 'NCT-CRC-HE-100K'

    write_CLS_script(model_idxs=model_idxs,
                     data_augmentation_mode=3,
                     edge_size=384,
                     batch_size=batch_size,
                     lr=0.000007,
                     lrf=0.35,
                     enable_tensorboard=True,
                     test_enable_attention_check=True,
                     dataset_name=dataset_name,
                     dataroot='/root/autodl-tmp/datasets/',
                     model_path='/root/autodl-tmp/saved_models',
                     draw_root='/root/autodl-tmp/runs')

    # 正式实验的时候，后面还需要做各种MIL的消融实验
    # TODO 更多write_MIL_script
    # 其次摸索CLS+特定模型vit+不同数据增强 对比实验结果
    augmentation_names = ['Cutout', 'Mixup', 'CutMix', 'ResizeMix', 'SaliencyMix', 'FMix']
    write_CLS_AUG_script(model_idx='ViT',
                         augmentation_names=augmentation_names,
                         data_augmentation_mode=3,
                         edge_size=384,
                         batch_size=batch_size,
                         lr=0.000007,
                         lrf=0.35,
                         enable_tensorboard=True,
                         test_enable_attention_check=True,
                         test_aug_attention_check=True,
                         dataset_name=dataset_name,
                         dataroot='/root/autodl-tmp/datasets/',
                         model_path='/root/autodl-tmp/saved_models',
                         draw_root='/root/autodl-tmp/runs')

    # 最后摸索MIL+ViT的实验结果
    MIL_model_idxs = ['ViT', ]
    # MIL_structures ablations
    write_MIL_script(model_idxs=MIL_model_idxs,
                     data_augmentation_mode=3,
                     edge_size=384,
                     batch_size=batch_size,
                     patch_size=16,
                     lr=0.000007,
                     lrf=0.35,
                     enable_tensorboard=True,
                     test_enable_attention_check=False,
                     dataset_name=dataset_name,
                     dataroot='/root/autodl-tmp/datasets/',
                     model_path='/root/autodl-tmp/saved_models',
                     draw_root='/root/autodl-tmp/runs',
                     imaging_root='/root/autodl-tmp/imaging_results')
    write_MIL_script(model_idxs=MIL_model_idxs,
                     data_augmentation_mode=3,
                     edge_size=384,
                     batch_size=batch_size,
                     patch_size=64,
                     lr=0.000007,
                     lrf=0.35,
                     enable_tensorboard=True,
                     test_enable_attention_check=False,
                     dataset_name=dataset_name,
                     dataroot='/root/autodl-tmp/datasets/',
                     model_path='/root/autodl-tmp/saved_models',
                     draw_root='/root/autodl-tmp/runs',
                     imaging_root='/root/autodl-tmp/imaging_results')
    write_MIL_script(model_idxs=MIL_model_idxs,
                     data_augmentation_mode=3,
                     edge_size=384,
                     batch_size=batch_size,
                     patch_size=48,
                     lr=0.000007,
                     lrf=0.35,
                     enable_tensorboard=True,
                     test_enable_attention_check=False,
                     dataset_name=dataset_name,
                     dataroot='/root/autodl-tmp/datasets/',
                     model_path='/root/autodl-tmp/saved_models',
                     draw_root='/root/autodl-tmp/runs',
                     imaging_root='/root/autodl-tmp/imaging_results')
    write_MIL_script(model_idxs=MIL_model_idxs,
                     data_augmentation_mode=3,
                     edge_size=384,
                     batch_size=batch_size,
                     patch_size=96,
                     lr=0.000007,
                     lrf=0.35,
                     enable_tensorboard=True,
                     test_enable_attention_check=False,
                     dataset_name=dataset_name,
                     dataroot='/root/autodl-tmp/datasets/',
                     model_path='/root/autodl-tmp/saved_models',
                     draw_root='/root/autodl-tmp/runs',
                     imaging_root='/root/autodl-tmp/imaging_results')
    write_MIL_script(model_idxs=MIL_model_idxs,
                     data_augmentation_mode=3,
                     edge_size=384,
                     batch_size=batch_size,
                     patch_size=128,
                     lr=0.000007,
                     lrf=0.35,
                     enable_tensorboard=True,
                     test_enable_attention_check=False,
                     dataset_name=dataset_name,
                     dataroot='/root/autodl-tmp/datasets/',
                     model_path='/root/autodl-tmp/saved_models',
                     draw_root='/root/autodl-tmp/runs',
                     imaging_root='/root/autodl-tmp/imaging_results')

    # 调参实验的时候，先调MIL到最好，然后用参数去跑CLS实验看结果

    print('cd /home/pancreatic-cancer-diagnosis-tansformer/code/utils')
    print('')
    print(
        'python check_log_json.py --enable_notify --draw_root /root/autodl-tmp/runs --record_dir /root/autodl-tmp/CSV_logs')
    print('')
    print('shutdown')
'''


def get_args_parser():
    parser = argparse.ArgumentParser(description='Automatically write shell script for training')

    # Model Name or index
    parser.add_argument('--lr_mystr', default='401', type=str, help='Model lr EG: 506 -> 0.000006')
    parser.add_argument('--lrf_mystr', default='05', type=str, help='Model lrf EG: 05 -> cosine decay to 5%')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--edge_size', default=224, type=int, help='edge_size')

    parser.add_argument('--data_augmentation_mode', default=3, type=int, help='ROSE,pRCC:0; CAM16,WBC:3')
    parser.add_argument('--dataset_name', default='WBC', type=str, help='ROSE,pRCC,CAM16,WBC ?')

    parser.add_argument('--enable_tensorboard', action='store_true', help='enable tensorboard to save status')
    parser.add_argument('--test_enable_attention_check', action='store_true', help='enable Grad-CAM in test')
    parser.add_argument('--test_aug_attention_check', action='store_true',
                        help='enable additional augmented Grad-CAM check in test')
    parser.add_argument('--GPU_idx', default='0', type=str, help='Experiment GPU_idx EG: 0')

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    print('#!/bin/sh')
    print('')

    augmentation_names = ['Cutout', 'Mixup', 'CutMix', 'ResizeMix', 'SaliencyMix', 'FMix']
    write_CLS_AUG_script(model_idx='ViT',
                         augmentation_names=augmentation_names,
                         data_augmentation_mode=args.data_augmentation_mode,
                         edge_size=args.edge_size,
                         batch_size=args.batch_size,
                         lr=args.lr_mystr,
                         lrf=args.lrf_mystr,
                         enable_tensorboard=args.enable_tensorboard,
                         test_enable_attention_check=args.test_enable_attention_check,
                         test_aug_attention_check=args.test_aug_attention_check,
                         dataset_name=args.dataset_name,
                         dataroot='/root/autodl-tmp/datasets/',
                         model_path='/root/autodl-tmp/saved_models',
                         draw_root='/root/autodl-tmp/runs')

    '''
    # for PuzzleTuning
    # add MAE-CPIA
    write_additional_PuzzleTuning_comparison_script(add_idx='MAE-CPIA', lr_mystr=args.lr_mystr,
                                                    lrf_mystr=args.lrf_mystr,
                                                    data_augmentation_mode=args.data_augmentation_mode,
                                                    dataset_name=args.dataset_name, 
                                                    model_weight_idx='timm_mae_CPIAm_E100',
                                                    model_weight_name='ViT_b16_224_timm_mae_ALL_100.pth', 
                                                    GPU_idx=args.GPU_idx, Prompt_input=False)
    # add SDMAE-CPIA
    write_additional_PuzzleTuning_comparison_script(add_idx='SDMAE-CPIA', lr_mystr=args.lr_mystr,
                                                    lrf_mystr=args.lrf_mystr,
                                                    data_augmentation_mode=args.data_augmentation_mode,
                                                    dataset_name=args.dataset_name,
                                                    model_weight_idx='timm_SDMAE_CPIAm_E80',
                                                    model_weight_name='ViT_b16_224_timm_SDMAE_ALL_80.pth',
                                                    GPU_idx=args.GPU_idx, Prompt_input=False)
    # add GCMAE-CPIA
    write_additional_PuzzleTuning_comparison_script(add_idx='GCMAE-CPIA', lr_mystr=args.lr_mystr,
                                                    lrf_mystr=args.lrf_mystr,
                                                    data_augmentation_mode=args.data_augmentation_mode,
                                                    dataset_name=args.dataset_name,
                                                    model_weight_idx='timm_GCMAE_CPIAm_E80',
                                                    model_weight_name='ViT_b16_224_timm_GCMAE_ALL_80.pth',
                                                    GPU_idx=args.GPU_idx, Prompt_input=False)
    # add MAE+VPT
    write_additional_PuzzleTuning_comparison_script(add_idx='MAE-VPT_promptstate',
                                                    lr_mystr=args.lr_mystr,
                                                    lrf_mystr=args.lrf_mystr,
                                                    data_augmentation_mode=args.data_augmentation_mode,
                                                    dataset_name=args.dataset_name,
                                                    model_weight_idx='timm_mae_Prompt_CPIAm_E199_promptstate',
                                                    model_weight_name='ViT_b16_224_timm_PuzzleTuning_MAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth',
                                                    GPU_idx='0', Prompt_input=True)
    # add SAE-MAE-start
    write_additional_PuzzleTuning_comparison_script(add_idx='SAE-MAE-start_promptstate',
                                                    lr_mystr=args.lr_mystr,
                                                    lrf_mystr=args.lrf_mystr,
                                                    data_augmentation_mode=args.data_augmentation_mode,
                                                    dataset_name=args.dataset_name,
                                                    model_weight_idx='ViT_base_MAEImageNet_PuzzleTuning_SAE_E_199_promptstate',
                                                    model_weight_name='ViT_b16_224_MAEImageNet_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth',
                                                    GPU_idx='0', Prompt_input=True)
    # add SAE-Random-start
    write_additional_PuzzleTuning_comparison_script(add_idx='SAE-Random-start_promptstate',
                                                    lr_mystr=args.lr_mystr,
                                                    lrf_mystr=args.lrf_mystr,
                                                    data_augmentation_mode=args.data_augmentation_mode,
                                                    dataset_name=args.dataset_name,
                                                    model_weight_idx='ViT_base_Random_PuzzleTuning_SAE_E_199_promptstate',
                                                    model_weight_name='ViT_b16_224_Random_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth',
                                                    GPU_idx='0', Prompt_input=True)
    '''

    # rewrite all
    '''
    write_PuzzleTuning_comparison_script(lr_mystr=args.lr_mystr, lrf_mystr=args.lrf_mystr,
                                         data_augmentation_mode=args.data_augmentation_mode,
                                         dataset_name=args.dataset_name, GPU_idx=args.GPU_idx)
    '''


    '''
    NOTICE 
    we can use the following codes to generates the additional exp scripts
    
    # read and auto generate task info
    import os
    path='/root/autodl-tmp/PuzzleTuning_Comparison'
    data_augmentation_dic = {'ROSE': '0', 'pRCC': '0', 'CAM16': '3', 'WBC': '3'}
    for exp_root in os.listdir(path):
        out_sh_name = exp_root + '.sh'
        lr_mystr = exp_root.split('_')[0]
        lrf_mystr = exp_root.split('_')[1].split('lf')[-1]
        dataset_name = exp_root.split('_')[-1]
        data_augmentation_mode = data_augmentation_dic[dataset_name]
        print('nohup python Experiment_script_helper.py --lr_mystr ' + lr_mystr + ' --lrf_mystr ' + lrf_mystr
              + ' --data_augmentation_mode ' + data_augmentation_mode + ' --dataset_name ' + dataset_name + ' > '
              + out_sh_name + ' 2>&1 &')
    
    # then, we use the shell to run this code with the generated lines
    
    # the generate sh files has a nohup line at their first lines, so we can use this to erase
    remove_nohup_ignoring_input_at_first_line(directory='./')
    
    # we can use the func to combine the sh files:
    concatenate_the_lines_from_several_files(directory='./', cat_file='0.sh')
    '''