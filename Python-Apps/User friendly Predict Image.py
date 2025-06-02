import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import os
from ultralytics import YOLO
import threading
import time

class YOLOv8GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Application")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        self.font = ("Arial", 10)
        
        self.model = None
        self.model_name = tk.StringVar(value="yolov8n.pt")
        self.models = ["model.pt"]

        self.original_img = None
        self.processed_img = None
        self.photo_original = None
        self.photo_processed = None

        self.create_widgets()
        
    def create_widgets(self):
        top_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        
        model_label = tk.Label(top_frame, text="Select model:", font=self.font, bg="#f0f0f0")
        model_label.pack(side=tk.LEFT, padx=5)
        
        model_combo = ttk.Combobox(top_frame, textvariable=self.model_name, values=self.models, width=15)
        model_combo.pack(side=tk.LEFT, padx=5)

        load_model_btn = tk.Button(top_frame, text="Load model", font=self.font, 
                                   command=self.load_model, bg="#4CAF50", fg="white",
                                   padx=10, pady=5)
        load_model_btn.pack(side=tk.LEFT, padx=10)

        select_btn = tk.Button(top_frame, text="New image...", font=self.font, 
                              command=self.select_image, bg="#2196F3", fg="white",
                              padx=10, pady=5)
        select_btn.pack(side=tk.LEFT, padx=10)
        
        process_btn = tk.Button(top_frame, text="Call YOLOv8", font=self.font, 
                               command=self.process_image, bg="#FFC107", fg="black",
                               padx=10, pady=5)
        process_btn.pack(side=tk.LEFT, padx=10)
        
        save_btn = tk.Button(top_frame, text="Store result", font=self.font, 
                            command=self.save_image, bg="#FF5722", fg="white",
                            padx=10, pady=5, state=tk.DISABLED)
        self.save_btn = save_btn
        save_btn.pack(side=tk.LEFT, padx=10)
        
        self.status_var = tk.StringVar(value="You must select a model first")
        status_label = tk.Label(top_frame, textvariable=self.status_var, font=self.font, 
                               bg="#f0f0f0", fg="red")
        status_label.pack(side=tk.RIGHT, padx=20)

        middle_frame = tk.Frame(self.root, bg="#e0e0e0", padx=10, pady=10)
        middle_frame.pack(fill=tk.BOTH, expand=True)

        original_frame = tk.LabelFrame(middle_frame, text="Display image", font=self.font, 
                                      bg="#e0e0e0", padx=5, pady=5)
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.original_label = tk.Label(original_frame, bg="#ffffff")
        self.original_label.pack(fill=tk.BOTH, expand=True)

        processed_frame = tk.LabelFrame(middle_frame, text="Output", font=self.font, 
                                       bg="#e0e0e0", padx=5, pady=5)
        processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.processed_label = tk.Label(processed_frame, bg="#ffffff")
        self.processed_label.pack(fill=tk.BOTH, expand=True)
        
        bottom_frame = tk.LabelFrame(self.root, text="Detection result", font=self.font, 
                                    bg="#f0f0f0", padx=10, pady=10)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.info_text = tk.Text(bottom_frame, height=5, width=80, font=self.font, 
                                wrap=tk.WORD)
        self.info_text.pack(fill=tk.X)
        
    def load_model(self):
        model_name = self.model_name.get()
        self.status_var.set(f"Loading model: {model_name}")
        self.root.update()
        
        try:
            start_time = time.time()
            self.model = YOLO(model_name)
            elapsed_time = time.time() - start_time
            self.status_var.set(f"The model is now loaded. Time consumed: ({elapsed_time:.2f} second)")
            messagebox.showinfo("SUCCESS", f"Model {model_name} is now loaded")
        except Exception as e:
            self.status_var.set(f"Fail to load model: {str(e)}")
            messagebox.showerror("ERROR", f"Fail to load the model: {str(e)}")
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="New image...",
            filetypes=[("Valid image", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_original_image()
            self.status_var.set(f"Image selected: {os.path.basename(file_path)}")
            self.info_text.delete(1.0, tk.END)
            self.save_btn.config(state=tk.DISABLED)
    
    def display_original_image(self):
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.original_img = img

        img = self.resize_image(img)
        self.photo_original = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.original_label.config(image=self.photo_original)
    
    def process_image(self):
        if not hasattr(self, 'image_path'):
            messagebox.showwarning("ERROR", "You must selected an image first")
            return
        
        if self.model is None:
            messagebox.showwarning("ERROR", "You must load the model first")
            return
        
        self.status_var.set("Calling YOLOv8 model...")
        self.root.update()
        threading.Thread(target=self._process_image_thread).start()
    
    def _process_image_thread(self):
        try:
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = self.model(img, conf=0.25)
            
            for result in results:
                annotated_img = result.plot()
                self.processed_img = annotated_img

                self.display_detection_info(result)

            self.root.after(0, self.display_processed_image)
            self.status_var.set("Detection complete")
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Unable to proceed: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("ERROR", f"Fail to proceed the detection: {str(e)}"))
    
    def display_processed_image(self):
        if self.processed_img is not None:
            img = self.resize_image(self.processed_img)
            self.photo_processed = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.processed_label.config(image=self.photo_processed)
    
    def resize_image(self, img):
        max_width = self.original_label.winfo_width() - 20
        max_height = self.original_label.winfo_height() - 20
        
        if max_width <= 0 or max_height <= 0:
            max_width = 400
            max_height = 400
        
        height, width = img.shape[:2]

        ratio = min(max_width / width, max_height / height)

        if ratio >= 1:
            return img

        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return cv2.resize(img, (new_width, new_height))
    
    def display_detection_info(self, result):
        self.info_text.delete(1.0, tk.END)
        boxes = result.boxes.cpu().numpy()
        self.info_text.insert(tk.END, f"Number of objects find {len(boxes)}:\n")

        class_counts = {}
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

        for class_name, count in sorted(class_counts.items()):
            self.info_text.insert(tk.END, f"- {class_name}: {count}个\n")
    
    def save_image(self):
        if self.processed_img is None:
            messagebox.showwarning("ERROR", "You must run the detection first")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Store output",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg;*.jpeg")]
        )
        
        if file_path:
            try:
                img = cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, img)
                messagebox.showinfo("SUCCESS", f"Image has been stored to: {file_path}")
            except Exception as e:
                messagebox.showerror("ERROR", f"Unable to store the image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv8GUI(root)
    root.mainloop()