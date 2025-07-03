class TrainingConfig:
    # 模型参数
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    task_type = "SEQ_2_SEQ_LM"
    target_modules = ["q_proj", "v_proj"]   # 微调Wq,Wv的效果和微调Wq,Wk,Wv,Wo的效果相似

    # 训练参数
    batch_size = 1
    learning_rate = 3e-4
    mixed_precision = True          # 混合精度训练
    num_epochs = 1
    max_length = 512
    save_dir = "./checkpoints"
    
    # 数据路径
    data_root = "./data"
    question_dir = "train_question"
    label_dir = "train_label"
    # cot数据路径
    cot_data_dir = "train_cot_data"
    cot_data_file = 'train.json'
    full_cot_data_file = data_root + '/' + cot_data_dir + '/' + cot_data_file
    # 预测数据路径
    predict_file = 'data_query.txt'

    # 早停参数配置
    early_stop_patience = 3  # 允许连续不改进的步数间隔
    min_delta = 1e-4         # 损失改进阈值
    no_improve_steps = 0     # 连续未改进计数

    # 最优模型
    best_model = save_dir+'/best_model_cot'