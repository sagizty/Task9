'''
Script  ver： Nov 26th 14:00
This is for adding a new comparison methods to existing comparison experiments folder
'''

import os

def move_CLS_exp(now_path_all,correct_all_path):
    for path in os.listdir(now_path_all):
        if path.split('_')[-1] == 'CLS':
            dataset_name = path.split('_')[-2]
            lrf_name = path.split('_')[-4]
            lr_name = path.split('_')[-5]
            new_path = lr_name + '_' + lrf_name + '_' + dataset_name

            print('mv ' + os.path.join(now_path_all, path) + ' ' + os.path.join(correct_all_path, new_path))
            print('mv ' + os.path.join(now_path_all, path + '_test') + ' ' + os.path.join(correct_all_path, new_path))


def move_PuzzleTuning_comp_exp(now_path_all,correct_all_path):
    for folder_name in os.listdir(now_path_all):
        print('mv ' + os.path.join(now_path_all, folder_name) + '/* ' + os.path.join(correct_all_path, folder_name))


if __name__ == '__main__':
    now_path_all = '/Users/zhangtianyi/Downloads/non/PuzzleTuning_Comparison_mae'
    correct_all_path = '/Users/zhangtianyi/Downloads/non/PuzzleTuning_Comparison'
    move_PuzzleTuning_comp_exp(now_path_all, correct_all_path)
