/
│
├── /deeplearning                           # This folder contains nnunet, resnet, and dataset, and data eda files.
│   ├── /Datasets                      # dataset
│   │   ├── /imagesTrRgb               # training images
│   │   ├── /imagesTs                  # test image
│   │   ├── /labelsTr                  # training labels
│   │   ├── /labelsTs                  # test labels
│   │   ├── /sam_finetune_output       # SAM fine tune prediction output (small model + 0.1m training data)
│   │   ├── /sam_finetune_output_0.5M  # SAM fine tune prediction output (small model + 0.5m training data)
│   │   └── /sam_finetune_output_1M    # SAM fine tune prediction output (small model + 1m training data)
│   ├── /eda                           # raw images exploration analysis
│   ├── /nnUNet                        # nnUNet downloaded from github: https://github.com/MIC-DKFZ/nnUNet
│   ├── /nnUNet_preprocessed           # nnUNet preprocessed dataset, coming from nnUNet_raw
│   ├── /nnUNet_raw                    # nnUNet raw dataset 002, 003, 004 respective preprocessed data, and result are under nnUNet_preprocessed 002, 003, 004, and nnUNet_result  002, 003, 004
│   ├── /nnUNet_result                 # nnUNet result of dataset 002, 003, 004
│   └── /resnet_tumor                  # resnet data preprocess, model code, etc
│
└── /SAM/segment-anything-2                # segment-anything-2 model
    ├── /model_0.5m_small.torch        # small model + 0.5m training iteration
    ├── /model_1m_small.torch          # small model + 1m training iteration
    ├── /model_large.torch             # large model + 0.1m training iteration
    ├── /model.torch                   # small model + 0.1m training iteration
    ├── loss_plot.png                  # training loss visualization
    ├── annotation.png                 # tumor prediction area
    ├── annotation_white_label.png     # tumor label
    ├── prediction_all.py              # fine-tuned model prediction script
    └── train.py                       # fine-tune training script
    ...other SAM files downloaded from github: https://github.com/facebookresearch/sam2
    
          

