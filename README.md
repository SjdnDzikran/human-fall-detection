# Human Falling Detection and Tracking

> **Note:** This repository is a modified version based on the original work by gajuuzz: [https://github.com/gajuuzz/gajuuzz-human-falling-detect-tracks](https://github.com/gajuuzz/gajuuzz-human-falling-detect-tracks).
>
> The pre-trained models required for execution can be downloaded from this Google Drive link: [https://drive.google.com/drive/folders/1lrTI56k9QiIfMJhG9kzNjBzJh98KCIIO](https://drive.google.com/drive/folders/1lrTI56k9QiIfMJhG9kzNjBzJh98KCIIO). Please refer to the setup instructions below.

Using Tiny-YOLO oneclass to detect each person in the frame and use
[AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to get skeleton-pose and then use
[ST-GCN](https://github.com/yysijie/st-gcn) based model (TSSTG) to predict action from every 30 frames
of each person tracks.

Which now support 7 actions: Standing, Walking, Sitting, Lying Down, Stand up, Sit down, Fall Down.

<div align="center">
    <img src="sample1.gif" width="416">
</div>

## Prerequisites

- Python > 3.6
- Pytorch > 1.3.1
- OpenCV (`opencv-python-headless` recommended for servers/Colab)
- `gdown` (for downloading models from Google Drive in Colab setup)
- `screeninfo`, `tk` (Note: GUI features in `App.py` may not work in all environments like Colab)

Original test run on: i7-8750H CPU @ 2.20GHz x12, GeForce RTX 2070 8GB, CUDA 10.2

## Data

This project utilizes pre-trained models. The original models were trained as follows:

*   **Person Detector (Tiny-YOLO oneclass):** Trained on a rotation-augmented [COCO](http://cocodataset.org/#home) person keypoints dataset for robust person detection.
*   **Action Recognition (TSSTG):** Trained using data from the [Le2i](http://le2i.cnrs.fr/Fall-detection-Dataset?lang=fr) Fall detection Dataset (Coffee room, Home). Skeleton poses were extracted using AlphaPose and action frames were labeled manually.

**Note:** This repository does **not** contain the raw training datasets (COCO images or Le2i videos). The `Data/` directory contains scripts used for processing such raw data during the original training phase. For running inference (`main.py`), you only need the pre-trained models and your own input video.

## Pre-Trained Models

This project relies on several pre-trained models for detection, pose estimation, and action recognition. The necessary model weights **must** be downloaded from the following Google Drive folder:

*   **Models Link:** [https://drive.google.com/drive/folders/1lrTI56k9QiIfMJhG9kzNjBzJh98KCIIO](https://drive.google.com/drive/folders/1lrTI56k9QiIfMJhG9kzNjBzJh98KCIIO)

Please follow the instructions in the **Google Colab Setup & Inference** section below to download and place these models correctly within the `./Models/` directory structure. The specific models included are:

*   Tiny-YOLO oneclass (person detector)
*   SPPE FastPose (AlphaPose variant for pose estimation - ResNet50 and/or ResNet101 based)
*   TSSTG (Two-Stream Spatio-Temporal Graph network for action recognition)

## Google Colab Setup & Inference

Follow these steps to run the fall detection inference (`main.py`) on a video file within a Google Colab notebook:

1.  **Clone this Repository:**
    ```bash
    # Clone your specific fork/clone of the repository
    !git clone [URL_OF_YOUR_GITHUB_REPOSITORY]
    %cd [NAME_OF_YOUR_CLONED_FOLDER]
    ```

2.  **Install Dependencies:**
    ```bash
    !pip install torch torchvision torchaudio opencv-python-headless gdown screeninfo tk -q
    # Note: Specific versions might be needed if errors occur.
    ```

3.  **Create Model Directories:**
    ```bash
    !mkdir -p Models/yolo-tiny-onecls Models/sppe Models/TSSTG
    ```

4.  **Download Pre-trained Models:**
    Download the required models from the Google Drive link provided above using `gdown`. You will need the unique **File ID** for each model file from the shared Google Drive folder.
    *(How to get File ID: Open the Drive folder link, right-click on a file, select "Get link", and copy the ID part from the URL like `.../d/FILE_ID/view?usp=sharing`)*

    ```bash
    # --- Replace FILE_ID_... with the actual IDs from the Google Drive Link ---

    # TinyYOLOv3 One-Class Detector
    !gdown --id FILE_ID_YOLO_CFG -O Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg
    !gdown --id FILE_ID_YOLO_PTH -O Models/yolo-tiny-onecls/best-model.pth

    # SPPE Pose Estimator (Download the one matching main.py's default or your choice)
    # Example for ResNet50 (check main.py arguments/defaults)
    !gdown --id FILE_ID_SPPE_RES50 -O Models/sppe/fast_res50_256x192.pth
    # Example for ResNet101 (Uncomment if needed)
    # !gdown --id FILE_ID_SPPE_RES101 -O Models/sppe/fast_res101_320x256.pth

    # TSSTG Action Recognizer
    !gdown --id FILE_ID_TSSTG -O Models/TSSTG/tsstg-model.pth

    # --- Verify files are downloaded in the correct folders ---
    !ls -R Models
    ```

5.  **Upload Test Video:**
    Use the file browser panel on the left side of Colab to upload your test video file (e.g., `my_test_video.mp4`) into the main directory of the cloned repository (`[NAME_OF_YOUR_CLONED_FOLDER]`).

6.  **Run Inference:**
    Execute the `main.py` script. Adjust parameters as needed.
    ```bash
    !python main.py --camera my_test_video.mp4 --device cuda --save_out output.avi --show_skeleton True
    # --camera: Path to your uploaded video file
    # --device: Use 'cuda' if GPU runtime is enabled, otherwise 'cpu'
    # --save_out: Optional: Saves the processed video with overlays to 'output.avi'
    # --show_skeleton: Set to True (default) or False to control skeleton overlay
    # --detection_input_size: e.g., 384 (default) or 416. Must match YOLO model if changed.
    # --pose_input_size: e.g., '224x160' (default) or '256x192'. Must match SPPE model used.
    # --pose_backbone: e.g., 'resnet50' (default) or 'resnet101'. Must match SPPE model used.
    ```

7.  **Check Output:**
    If you used `--save_out output.avi`, the processed video will appear in the file browser. You can download it to view the results. Check the Colab cell output for any printed messages or errors.

## Basic Use (Original Setup)

1.  Ensure all prerequisites are installed.
2.  Download all pre-trained models from the Google Drive link provided above into the respective `./Models` subfolders.
3.  Run `main.py`:
    ```bash
    python main.py ${video_file_path or camera_source_id} [options]
    ```
    Example: `python main.py ../Data/falldata/Home/Videos/video (1).avi`

    **Note:** For a guided setup, especially within Google Colab, please refer to the **Google Colab Setup & Inference** section above.

## Reference

-   Original Base Repository: [gajuuzz/gajuuzz-human-falling-detect-tracks](https://github.com/gajuuzz/gajuuzz-human-falling-detect-tracks)
-   AlphaPose Fork Used: [Amanbhandula/AlphaPose](https://github.com/Amanbhandula/AlphaPose) (Link from original repo)
-   ST-GCN Paper/Repo: [yysijie/st-gcn](https://github.com/yysijie/st-gcn)