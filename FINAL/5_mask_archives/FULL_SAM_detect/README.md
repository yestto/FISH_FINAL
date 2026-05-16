# FULL SAM Detection Pipeline

This folder contains a complete, reusable pipeline to extract perfect fish segmentations from any frames you provide.

## Folder Structure
- `input_frames/` : **Drop the images you want to process here.** You can put `.jpg` or `.png` files directly inside this folder.
- `output_frames/` : The pipeline will automatically save the processed images here. They will show the original image with the SAM masks and YOLO bounding boxes drawn over the top.
- `run_full_sam_pipeline.py` : The Python script that runs the pipeline.

## How it works
When you run the script, it does the following:
1. Loads your custom YOLOv8 model to locate the fish in the frame and draw a bounding box.
2. Applies a CLAHE contrast enhancement algorithm behind-the-scenes to deal with murky water.
3. Passes the enhanced image and the YOLO bounding box to the Meta **Full Segment Anything Model (ViT-B)**.
4. SAM predicts the precise silhouette of the fish.
5. Applies the "gentle fin removal" (Morphological Opening) technique to clean up the mask.
6. Draws the semi-transparent mask and bounding box on the original image and saves it to the `output_frames` folder.

## How to use
1. Copy or drag-and-drop the frames you want to test into the `input_frames` folder.
2. Open your terminal/command prompt.
3. Run the script:
   ```bash
   python "run_full_sam_pipeline.py"
   ```
4. Open the `output_frames` folder to see the results!
