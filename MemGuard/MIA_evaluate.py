import os
import numpy as np
import math
import sys
import urllib
import pickle
import input_data_class
import argparse
sys.path.append('../')
from membership_inference_attacks import black_box_benchmarks
from privacy_risk_score_utils import calculate_risk_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run membership inference attacks')
    parser.add_argument('--dataset', type=str, default='location', help='location or texas')
    parser.add_argument('--predictions-dir', type=str, default='./saved_predictions', help='directory of saved predictions')
    parser.add_argument('--defended', type=int, default=1, help='1 means defended; 0 means natural')
    args = parser.parse_args()
    
    dataset = args.dataset
    input_data=input_data_class.InputData(dataset=dataset) # 读取数据集配置（数据集位置位置、各数据子集的索引位置）生成数据对象
    (x_target,y_target,l_target) =input_data.input_data_attacker_evaluate() # 从数据集中抽取用于攻击的数据
    npz_data = np.load('./saved_predictions/'+dataset+'_target_predictions.npz') # 预测结果为每个标签的置信度
    if args.defended==1:
        target_predictions = npz_data['defense_output']
    else:
        target_predictions = npz_data['tc_output']
    # l_target==1表示训练集，l_target==0表示测试集
    target_train_performance = (target_predictions[l_target==1], y_target[l_target==1].astype('int32'))
    target_test_performance = (target_predictions[l_target==0], y_target[l_target==0].astype('int32'))

    (x_shadow,y_shadow,l_shadow) =input_data.input_data_attacker_adv1()
    npz_data = np.load('./saved_predictions/'+dataset+'_shadow_predictions.npz')
    if args.defended==1:
        shadow_predictions = npz_data['defense_output']
    else:
        shadow_predictions = npz_data['tc_output']
    # 影子模型的训练集和测试集
    shadow_train_performance = (shadow_predictions[l_shadow==1], y_shadow[l_shadow==1].astype('int32'))
    shadow_test_performance = (shadow_predictions[l_shadow==0], y_shadow[l_shadow==0].astype('int32'))
    
    print('Perform membership inference attacks!!!')
    if args.dataset=='location':
        num_classes = 30
    else:
        num_classes = 100
    MIA = black_box_benchmarks(shadow_train_performance,shadow_test_performance,
                         target_train_performance,target_test_performance,num_classes=num_classes)
    MIA._mem_inf_benchmarks()

    risk_score = calculate_risk_score(MIA.s_tr_m_entr, MIA.s_te_m_entr, MIA.s_tr_labels, MIA.s_te_labels, MIA.t_tr_m_entr, MIA.t_tr_labels)

    print("Risk score: ", np.average(risk_score))