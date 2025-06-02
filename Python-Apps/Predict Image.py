from ultralytics import YOLO
import cv2
import os

def process_image(image_path="image.png", output_path="output.png", model_name="model.pt"):

    if not os.path.exists(image_path):
        print(f"[ERROR] The image does not find: {image_path}")
        return

    model = YOLO(model_name)

    image = cv2.imread(image_path)

    results = model(image, conf=0.1)
    
    for result in results:
        annotated_image = result.plot()
    
    cv2.imwrite(output_path, annotated_image)
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        print(f" There are {len(boxes)} Object has been detected:")
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"- {class_name} (Confidence interval: {confidence:.2f})")
    
    print(f"The output has been stored to: {output_path}")

if __name__ == "__main__":
    process_image()