Current folder has:
- nnunet result summary json file
- resnet preprocessing and training files;
- sam training and predicting files
All files and predcition result images, and models, and checkpoints(large files) are available at: https://virginiatech-my.sharepoint.com/:f:/g/personal/ran22_vt_edu/EhyHz95Y4_hGlKvhzvFOe9sBZ-zUtiGTLWY84pqZ_P5t4Q?e=kpIoOM

/
│
├── /deeplearning                            # This folder contains nnUNet, ResNet, datasets, and EDA files.                          
│   ├── /Datasets                            # Dataset folder                                                                      
│   │   ├── /imagesTrRgb                     # Training images                                                                     
│   │   ├── /imagesTs                        # Test images                                                                         
│   │   ├── /labelsTr                        # Training labels                                                                     
│   │   ├── /labelsTs                        # Test labels                                                                         
│   │   ├── /sam_finetune_output             # SAM fine-tune prediction output (small model + 0.1m training data)                  
│   │   ├── /sam_finetune_output_0.5M        # SAM fine-tune prediction output (small model + 0.5m training data)                  
│   │   └── /sam_finetune_output_1M          # SAM fine-tune prediction output (small model + 1m training data)                   
│   ├── /eda                                 # Raw images exploration and analysis                                                 
│   ├── /nnUNet                              # nnUNet downloaded from GitHub                                                       
│   │   └── https://github.com/MIC-DKFZ/nnUNet                                                                                     
│   ├── /nnUNet_preprocessed                 # Preprocessed datasets from nnUNet_raw                                               
│   ├── /nnUNet_raw                          # Raw datasets (002, 003, 004) with respective preprocessed and result data           
│   ├── /nnUNet_result                       # Results for datasets (002, 003, 004)                                                
│   └── /resnet_tumor                        # ResNet data preprocessing, model code, etc.                                         
│
└── /SAM/segment-anything-2                  # segment-anything-2 model folder                                                     
    ├── /model_0.5m_small.torch              # Small model + 0.5m training iterations                                              
    ├── /model_1m_small.torch                # Small model + 1m training iterations                                                
    ├── /model_large.torch                   # Large model + 0.1m training iterations                                              
    ├── /model.torch                         # Small model + 0.1m training iterations                                              
    ├── loss_plot.png                        # Training loss visualization                                                         
    ├── annotation.png                       # Tumor prediction visualization                                                      
    ├── annotation_white_label.png           # Tumor label visualization                                                           
    ├── prediction_all.py                    # Fine-tuned model prediction script                                                  
    └── train.py                             # Fine-tune training script                                                           
        ...other SAM files downloaded from: https://github.com/facebookresearch/sam2                                                           
        
