import gradio as gr
import cv2
import requests
import os
from collections import deque

from ultralytics import YOLO

file_urls = [
    'https://huggingface.co/spaces/iamsuman/ripe-and-unripe-tomatoes-detection/resolve/main/samples/riped_tomato_93.jpeg?download=true',
    'https://huggingface.co/spaces/iamsuman/ripe-and-unripe-tomatoes-detection/resolve/main/samples/unriped_tomato_18.jpeg?download=true',
    'https://huggingface.co/spaces/iamsuman/ripe-and-unripe-tomatoes-detection/resolve/main/samples/tomatoes.mp4?download=true',
]

def download_file(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)

for i, url in enumerate(file_urls):
    if 'mp4' in file_urls[i]:
        download_file(
            file_urls[i],
            f"video.mp4"
        )
    else:
        download_file(
            file_urls[i],
            f"image_{i}.jpg"
        )

model = YOLO('best.pt')
path  = [['image_0.jpg'], ['image_1.jpg']]
video_path = [['video.mp4']]




def show_preds_image(image_path):
    image = cv2.imread(image_path)
    outputs = model.predict(source=image_path)
    results = outputs[0].cpu().numpy()

    # Print the detected objects' information (class, coordinates, and probability)
    box = results[0].boxes
    names = model.model.names
    boxes = results.boxes

    for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):

        x1, y1, x2, y2 = map(int, box)

        class_name = names[int(cls)]
        print(class_name, "class_name", class_name.lower() == 'ripe')
        if class_name.lower() == 'ripe':
            color = (0, 0, 255)  # Red for ripe
        else:
            color = (0, 255, 0)  # Green for unripe

        # Draw rectangle around object
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # Display class label on top of rectangle
        label = f"{class_name.capitalize()}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color,  # Use the same color as the rectangle
            2,
            cv2.LINE_AA)
        
    # Convert image to RGB (Gradio expects RGB format)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

inputs_image = [
    gr.components.Image(type="filepath", label="Input Image"),
]
outputs_image = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Ripe And Unripe Tomatoes Detection",
    examples=path,
    cache_examples=False,
)

def show_preds_video_batch_centered(video_path, batch_size=16, iou_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    names = model.model.names  # cache class names

    # For IoU-based tracking of unique tomatoes
    unique_objects = {}  # id -> (class_name, last_box)
    next_id = 0
    total_ripe, total_unripe = 0, 0

    frame_buffer = deque()

    def compute_iou(box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def match_or_register_object(cls_name, box):
        nonlocal next_id, total_ripe, total_unripe
        # Try to match existing object by IoU
        for obj_id, (existing_cls, existing_box) in unique_objects.items():
            if compute_iou(existing_box, box) > iou_threshold:
                unique_objects[obj_id] = (cls_name, box)
                return obj_id
        # Register as new object
        unique_objects[next_id] = (cls_name, box)
        if cls_name.lower() == "ripe":
            total_ripe += 1
        else:
            total_unripe += 1
        next_id += 1
        return next_id - 1

    def process_batch(frames, results):
        for frame, output in zip(frames, results):
            current_ripe, current_unripe = set(), set()

            if output.boxes:
                boxes = output.boxes
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                for box, cls_id in zip(boxes.xyxy, cls_ids):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = names[cls_id]

                    obj_id = match_or_register_object(class_name, (x1, y1, x2, y2))

                    if class_name.lower() == "ripe":
                        current_ripe.add(obj_id)
                        color = (0, 0, 255)
                    else:
                        current_unripe.add(obj_id)
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name.capitalize()} ID:{obj_id}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Centered current counts ---
            current_text = f"Current → Ripe: {len(current_ripe)} | Unripe: {len(current_unripe)}"
            (text_w, _), _ = cv2.getTextSize(current_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = (frame_width - text_w) // 2
            cv2.putText(frame, current_text, (text_x, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # --- Centered total counts ---
            total_text = f"Total Seen → Ripe: {total_ripe} | Unripe: {total_unripe}"
            (text_w, _), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = (frame_width - text_w) // 2
            cv2.putText(frame, total_text, (text_x, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)

            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Read and process in batches ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)

        if len(frame_buffer) == batch_size:
            results = model.track(source=list(frame_buffer), persist=True, tracker="bytetrack.yaml", verbose=False)
            yield from process_batch(frame_buffer, results)
            frame_buffer.clear()

    if frame_buffer:
        results = model.track(source=list(frame_buffer), persist=True, tracker="bytetrack.yaml", verbose=False)
        yield from process_batch(frame_buffer, results)

    cap.release()
    print(f"Final Totals → Ripe: {total_ripe}, Unripe: {total_unripe}")

# def show_preds_video(video_path):
#     results = model.track(source=video_path, persist=True, tracker="bytetrack.yaml", verbose=False, stream=True)
    
#     ripe_ids = set()
#     unripe_ids = set()
    
#     # Get video frame dimensions for centering text
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     cap.release()

#     for output in results:
#         frame = output.orig_img
        
#         if output.boxes and output.boxes.id is not None:
#             names = model.model.names
#             boxes = output.boxes
#             ids = boxes.id.cpu().numpy().astype(int)
#             classes = boxes.cls.cpu().numpy().astype(int)

#             for box, cls, track_id in zip(boxes.xyxy, classes, ids):
#                 x1, y1, x2, y2 = map(int, box)
#                 class_name = names[cls]

#                 # Define BGR colors directly for OpenCV functions
#                 if class_name.lower() == "ripe":
#                     # To get RED in Gradio (RGB), you need to use (255, 0, 0) BGR
#                     # Note: You were using (0, 0, 255) which is Blue in RGB after conversion.
#                     color = (0, 0, 255)
#                     ripe_ids.add(track_id)
#                 else:
#                     # To get GREEN in Gradio (RGB), you need to use (0, 255, 0) BGR.
#                     # This color is already correct.
#                     color = (0, 255, 0)
#                     unripe_ids.add(track_id)

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, f"{class_name.capitalize()} ID:{track_id}",
#                             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         ripe_count_text = f"Ripe: {len(ripe_ids)}"
#         unripe_count_text = f"Unripe: {len(unripe_ids)}"
#         full_text = f"{ripe_count_text} | {unripe_count_text}"

#         # Get text size to center it
#         (text_width, text_height), baseline = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#         text_x = (frame_width - text_width) // 2
#         text_y = 40 # A fixed position at the top

#         # Display the counts at the top center
#         cv2.putText(frame, full_text, (text_x, text_y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
#         # This line is crucial for the fix.
#         # It correctly converts the frame from BGR to RGB for Gradio.
#         yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     print(f"Final Counts → Ripe: {len(ripe_ids)}, Unripe: {len(unripe_ids)}")



inputs_video = [
    gr.components.Video(label="Input Video"),

]
outputs_video = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_video = gr.Interface(
    fn=show_preds_video_batch_centered,
    inputs=inputs_video,
    outputs=outputs_video,
    title="Ripe And Unripe Tomatoes Detection",
    examples=video_path,
    cache_examples=False,
)

gr.TabbedInterface(
    [interface_image, interface_video],
    tab_names=['Image inference', 'Video inference']
).queue().launch(share=True)