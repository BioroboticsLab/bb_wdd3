## Enhancing Automated Waggle Dance Detection: Evaluation and Optimization of Existing Methodologies for Improved Accuracy and Robustness

This tool was developed as part of the Thesis work by Prajna Narayan Bhat under the supervision of Prof. Tim Landgraf. 

This repository contains the code and experiments that focus on video-based object detection and temporal feature modeling to detect and decode honey bee waggle dances. 
The work explores advanced deep learning architectures for accurate detection and motion analysis across video frames, integrating backbone networks with temporal modules to effectively capture dynamic patterns.
It primarily addresses the limitations of the existing waggle dance detector - wdd2 (https://github.com/BioroboticsLab/bb_wdd2/tree/master).

The directory structure is as follows: 

.\
├── **data** \
│   ├── videos \
│   ├── annotations \
│   ├── eval_videos \
│   └── eval_annotations \
├── **experiments** \
│   ├── logs \
│   ├── checkpoints \
│   └── evaluation_results \
├── **notebooks** \
├── **scripts** (Starting points to run preprocessing, training, evaluation, and generate visualisations) \
│   ├── eval.py \
│   ├── train.py \ 
│   ├── visualize_data.py \
│   └── sample_videos_and_annotations.py \
├── **src** \
│   ├── data \
│   ├── evaluation \
│   ├── losses \
│   ├── models \
│   ├── training \
│   └── utils \
├── **setup.py**\
└── **requirements.txt** 


### Setup & Installation

To run this project, it is recommended to create a virtual environment. You can use either Conda or a Python virtual environment.

1. Create and activate a virtual environment

Using Conda:
```
  conda create -n wdd3_env python=3.10 -y
  conda activate wdd3_env
```
Using Python env:
```
  python -m venv wdd3_env
  # On Windows
  wdd3_env\Scripts\activate
  # On macOS/Linux
  source wdd3_env/bin/activate
```

2. Install dependencies

Once the environment is activated, install the required packages:

```
  pip install -r requirements.txt
```

3. Install the package

To make the code accessible as a Python package, install it using setup.py:

```
  python setup.py install
```

After this, the environment is ready, and you can start running the scripts, notebooks, and experiments.

### Running Inference

Before running inference, you need to prepare your annotations and preprocess them.

1. Annotate videos

Use the annotation tool (https://github.com/BioroboticsLab/bb_waggledance_annotator) to create annotation files for the videos you want to evaluate.

2. Preprocess annotations

Run the preprocess_annotations.py script to expand or format your annotation files properly:

```
  preprocess_annotations --input_folder <path_to_raw_annotations> --output_folder <path_to_expanded_annotations>
```

3. Run inference

After preprocessing, you can run inference on your evaluation videos using the trained model:

```
  infer \
    --model_path "<path_to_trained_model>/best_model.pth" \
    --video_folder "<path_to_eval_videos>" \
    --gt_folder "<path_to_expanded_annotations>" \
    --output_folder "<path_to_eval_results>" \
    --confidence 0.99 \
    --stride 4 \
    --batch_size 32 \
    --pos-thresholds 50 \
    --iou-thresholds 0.5 \
    --angular-thresholds 40 \
    --consolidation-strategy line_nms \
    --experiment-name
```

Explanation of key arguments:
--confidence: Detection confidence threshold.
--stride: Frame stride for sliding window input clip generation.
--batch_size: Batch size for processing frames.
--pos-thresholds, --iou-thresholds, --angular-thresholds: Thresholds for post-processing detections.
--consolidation-strategy: Strategy for consolidating detections across frames.

You can also run ``` infer --help``` to get a more detailed version of the command line arguments and their description.

### Training (Fine-tune) 

1. Prepare your dataset

Ensure that your training videos and annotations are ready and properly preprocessed.

2. Run training

You can run the training script using the following command:

```
  train \
    --data-dir <path_to_videos> \
    --annotations <path_to_annotations_csv> \
    --batch-size 32 \
    --epochs 200 \
    --lr 3e-4 \
    --device cuda \
    --log-dir <path_to_logs> \
    --checkpoint-dir <path_to_checkpoints> \
    --experiment-name <name_of_experiment> \
    --resume <path_to_checkpoint.pth> \
    --epochs 200
```

Using the option --resume will continue training from the epoch stopped at the checkpoint. 

