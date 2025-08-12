import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import torch
from ultralytics import YOLO
import cv2
import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from pathlib import Path
from collections import Counter

class AdvancedLeafDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 Advanced Leaf Disease Detection System")
        self.root.geometry("1200x650+0+0")
        self.root.minsize(1200, 650)
        
        # Initialize variables
        self.img_path = None
        self.batch_images = []
        self.detection_history = []
        self.model = None
        self.video_path = None
        self.webcam_active = False
        self.cap = None
        
        # Load model
        self.load_model()
        
        # Setup UI
        self.setup_styles()
        self.create_ui()
        self.root.after(500,self.show_welcome_message)
        
    def show_welcome_message(self):
        welcome_text = (
        "🌿 Welcome to Advanced Leaf Disease Detection System 🌿\n\n"
        "📋 Instructions:\n"
        "1. Go to '🔍 Detection' tab to load images, videos, or start the webcam.\n"
        "2. Adjust brightness/contrast if needed before detection.\n"
        "3. Click 'Detect Disease' to start analysis.\n"
        "4. Save results or export reports as needed.\n\n"
        "✨ Features:\n"
        "- Single image, video, and live webcam disease detection\n"
        "- Batch processing of multiple images\n"
        "- Data analysis with interactive charts\n"
        "- Adjustable detection confidence & image enhancement\n\n"
        "💡 Tips:\n"
        "- Use high-quality, well-lit images for best results.\n"
        "- Adjust confidence threshold if detections are too many/too few.\n"
        "- Check 'Analysis' tab for statistical insights."
        "- View application at maximum size and resolution"
    )
        messagebox.showinfo("Welcome", welcome_text)

    def load_model(self):
        """Load YOLOv8 model with error handling"""
        MODEL_PATH = "best.pt"
        try:
            self.model = YOLO(MODEL_PATH)
            print("✅ Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Model Load Error", 
                               f"Could not load YOLO model: {e}\n\nPlease ensure 'best.pt' is in the same directory.")
            self.model = None
        
    
    def setup_styles(self):
        """Setup custom styles for ttk widgets with a colorful theme"""
        self.style = ttk.Style()
        
        # Define a cheerful, plant-themed color palette
        self.colorful_palette = {
            'bg': '#E8F5E9',  # Lightest green
            'fg': '#2E2E2E',  # Dark gray for text
            'select_bg': '#C8E6C9', # A slightly darker light green
            'select_fg': '#2E2E2E',
            'button_bg': '#4CAF50', # Vibrant green
            'button_fg': "#7C1280", # White text
            'accent': '#66BB6A', # Brighter green
            'border': '#A5D6A7', # Border color
            'heading_bg': '#388E3C', # Dark green for headings
            'heading_fg': "#78810C"
        }
        
        # Set main window background
        self.root.configure(bg=self.colorful_palette['bg'])
        
        # Configure styles
        self.style.configure('TFrame', background=self.colorful_palette['bg'])
        self.style.configure('TNotebook', background=self.colorful_palette['bg'], borderwidth=0)
        self.style.configure('TNotebook.Tab', background=self.colorful_palette['select_bg'], foreground=self.colorful_palette['fg'])
        self.style.map('TNotebook.Tab', background=[('selected', self.colorful_palette['accent'])])

        self.style.configure('TLabelFrame', background=self.colorful_palette['bg'], foreground=self.colorful_palette['fg'], 
                             relief='flat', bordercolor=self.colorful_palette['border'], padding=10)
        self.style.configure('TLabelFrame.Label', background=self.colorful_palette['bg'], foreground=self.colorful_palette['fg'], font=('Arial', 10, 'bold'))

        self.style.configure('TLabel', background=self.colorful_palette['bg'], foreground=self.colorful_palette['fg'])

        self.style.configure('TButton', background=self.colorful_palette['button_bg'], foreground=self.colorful_palette['button_fg'], 
                             font=('Arial', 10, 'bold'), relief='flat')
        self.style.map('TButton', background=[('active', self.colorful_palette['accent'])])

        self.style.configure('TEntry', fieldbackground=self.colorful_palette['select_bg'], foreground=self.colorful_palette['fg'], relief='flat')
        self.style.configure('TCombobox', fieldbackground=self.colorful_palette['select_bg'], foreground=self.colorful_palette['fg'], relief='flat')
        
        # Treeview styles
        self.style.configure('Treeview', background=self.colorful_palette['select_bg'], foreground=self.colorful_palette['fg'], fieldbackground=self.colorful_palette['select_bg'])
        self.style.configure('Treeview.Heading', font=('Arial', 10, 'bold'), background=self.colorful_palette['heading_bg'], foreground=self.colorful_palette['heading_fg'])
        self.style.map('Treeview.Heading', background=[('hover', self.colorful_palette['accent'])])

        # Progressbar style
        self.style.configure('TProgressbar', background=self.colorful_palette['button_bg'], troughcolor=self.colorful_palette['select_bg'])

    def create_ui(self):
        """Create the main user interface"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.create_detection_tab()
        self.create_batch_tab()
        self.create_analysis_tab()
        self.create_settings_tab()
        
        self.create_status_bar()
    
    def create_detection_tab(self):
        """Create single image detection tab"""
        self.detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.detection_frame, text="🔍 Detection")
        
        left_panel = ttk.Frame(self.detection_frame, width=300)
        left_panel.pack(side='left', fill='y', padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        controls_frame = ttk.LabelFrame(left_panel, text="Input Controls", padding=5)
        controls_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Button(controls_frame, text="📂 Load Image", 
                  command=self.load_image).pack(fill='x', pady=2)
        ttk.Button(controls_frame, text="📹 Load Video", 
                  command=self.load_video).pack(fill='x', pady=2)
        ttk.Button(controls_frame, text="🎥 Start Webcam", 
                  command=self.start_webcam).pack(fill='x', pady=2)
        ttk.Button(controls_frame, text="🛑 Stop Webcam", 
                  command=self.stop_webcam).pack(fill='x', pady=2)
        ttk.Button(controls_frame, text="🔍 Detect Disease", 
                  command=self.detect_disease).pack(fill='x', pady=2)
        ttk.Button(controls_frame, text="💾 Save Results", 
                  command=self.save_results).pack(fill='x', pady=2)
        
        enhancement_frame = ttk.LabelFrame(left_panel, text="Enhancement", padding=5)
        enhancement_frame.pack(fill='x', pady=(0, 5))
        
        # Brightness
        ttk.Label(enhancement_frame, text="Brightness:").pack(anchor='w')
        self.brightness_var = tk.DoubleVar(value=1.0)
        brightness_scale = ttk.Scale(
            enhancement_frame, from_=0.5, to=2.0,
            variable=self.brightness_var, command=self.enhance_image
        )
        brightness_scale.pack(fill='x', pady=2)
        self.brightness_label = ttk.Label(enhancement_frame, text=f"{self.brightness_var.get():.2f}")
        self.brightness_label.pack()
        self.brightness_var.trace_add("write", lambda *args: self.brightness_label.config(text=f"{self.brightness_var.get():.2f}"))

        # Contrast
        ttk.Label(enhancement_frame, text="Contrast:").pack(anchor='w')
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_scale = ttk.Scale(
            enhancement_frame, from_=0.5, to=2.0,
            variable=self.contrast_var, command=self.enhance_image
        )
        contrast_scale.pack(fill='x', pady=2)
        self.contrast_label = ttk.Label(enhancement_frame, text=f"{self.contrast_var.get():.2f}")
        self.contrast_label.pack()
        self.contrast_var.trace_add("write", lambda *args: self.contrast_label.config(text=f"{self.contrast_var.get():.2f}"))
        
        ttk.Button(enhancement_frame, text="Reset", 
                  command=self.reset_enhancements).pack(fill='x', pady=2)
        
        params_frame = ttk.LabelFrame(left_panel, text="Parameters", padding=5)
        params_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Label(params_frame, text="Confidence Threshold:").pack(anchor='w')
        self.confidence_var = tk.DoubleVar(value=0.4)
        ttk.Scale(params_frame, from_=0.1, to=0.9, variable=self.confidence_var
        ).pack(fill='x', pady=2)
        self.confidence_label = ttk.Label(params_frame, text=f"{self.confidence_var.get():.2f}")
        self.confidence_label.pack()
        self.confidence_var.trace_add("write", lambda *args: self.confidence_label.config(text=f"{self.confidence_var.get():.2f}"))

        
        
        self.img_size_var = tk.IntVar(value=640)  # default YOLO size
        
        ttk.Button(params_frame, text="Reset", 
                  command=self.reset).pack(fill='x', pady=2)
        
        right_panel = ttk.Frame(self.detection_frame)
        right_panel.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.image_frame = ttk.LabelFrame(right_panel, text="Display", padding=5)
        self.image_frame.pack(fill='both', expand=True)
        
        self.image_canvas = tk.Canvas(self.image_frame, bg=self.colorful_palette['select_bg'], height=400)
        self.image_canvas.pack(fill='both', expand=True)
        
        results_frame = ttk.LabelFrame(right_panel, text="Results", padding=5)
        results_frame.pack(fill='x', pady=(5, 0))
        
        self.results_tree = ttk.Treeview(results_frame, columns=('Disease', 'Confidence', 'Location'), 
                                       show='headings', height=4)
        self.results_tree.heading('Disease', text='Disease')
        self.results_tree.heading('Confidence', text='Confidence')
        self.results_tree.heading('Location', text='Location')
        self.results_tree.pack(fill='x')
        
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
    
    def create_batch_tab(self):
        """Create batch processing tab"""
        self.batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_frame, text="📦 Batch Processing")
        
        controls_frame = ttk.LabelFrame(self.batch_frame, text="Batch Controls", padding=5)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(controls_frame, text="📁 Select Folder", 
                  command=self.select_batch_folder).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="▶️ Process Batch", 
                  command=self.process_batch).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="📊 Export Report", 
                  command=self.export_batch_report).pack(side='left', padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(controls_frame, variable=self.progress_var, 
                                          mode='determinate')
        self.progress_bar.pack(side='right', fill='x', expand=True, padx=10)
        
        batch_results_frame = ttk.LabelFrame(self.batch_frame, text="Batch Results", padding=5)
        batch_results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.batch_tree = ttk.Treeview(batch_results_frame, 
                                     columns=('File', 'Status', 'Diseases', 'Max_Confidence'), 
                                     show='headings')
        self.batch_tree.heading('File', text='File Name')
        self.batch_tree.heading('Status', text='Status')
        self.batch_tree.heading('Diseases', text='Diseases Found')
        self.batch_tree.heading('Max_Confidence', text='Max Confidence')
        self.batch_tree.pack(fill='both', expand=True)
        
        batch_scrollbar = ttk.Scrollbar(batch_results_frame, orient='vertical', 
                                      command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=batch_scrollbar.set)
        batch_scrollbar.pack(side='right', fill='y')
    
    def create_analysis_tab(self):
        """Create analysis and statistics tab"""
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="📈 Analysis")
        
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 5))
        
        self.fig.patch.set_facecolor(self.colorful_palette['bg'])
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor(self.colorful_palette['select_bg'])
            ax.tick_params(colors=self.colorful_palette['fg'], which='both')
            ax.yaxis.label.set_color(self.colorful_palette['fg'])
            ax.xaxis.label.set_color(self.colorful_palette['fg'])
            ax.title.set_color(self.colorful_palette['fg'])
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.analysis_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        analysis_controls = ttk.Frame(self.analysis_frame)
        analysis_controls.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(analysis_controls, text="🔄 Refresh Charts", 
                  command=self.update_charts).pack(side='left', padx=5)
        ttk.Button(analysis_controls, text="💾 Save Charts", 
                  command=self.save_charts).pack(side='left', padx=5)
        
        self.update_charts()
    
    def create_settings_tab(self):
        """Create settings and preferences tab"""
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="⚙️ Settings")
        
        model_frame = ttk.LabelFrame(self.settings_frame, text="Model Settings", padding=5)
        model_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(model_frame, text="Model Path:").pack(anchor='w')
        self.model_path_var = tk.StringVar(value="best.pt")
        ttk.Entry(model_frame, textvariable=self.model_path_var).pack(fill='x', pady=2)
        ttk.Button(model_frame, text="🔄 Reload Model", 
                  command=self.reload_model).pack(anchor='w', pady=2)
        
        export_frame = ttk.LabelFrame(self.settings_frame, text="Export Settings", padding=5)
        export_frame.pack(fill='x', padx=5, pady=5)
        
        self.save_images_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(export_frame, text="Save annotated images", 
                       variable=self.save_images_var).pack(anchor='w')
        
        self.save_json_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(export_frame, text="Save results as JSON", 
                       variable=self.save_json_var).pack(anchor='w')
        
        about_frame = ttk.LabelFrame(self.settings_frame, text="About", padding=5)
        about_frame.pack(fill='x', padx=5, pady=5)
        
        about_text = """Advanced Leaf Disease Detection System v2.0
Features: Single/batch processing, video/webcam processing, enhancement, analysis, themes
Built with YOLOv8 and modern UI components."""
        
        ttk.Label(about_frame, text=about_text, justify='left').pack(anchor='w')
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        self.status_frame = ttk.Frame(self.root, style='Status.TFrame')
        self.status_frame.pack(fill='x', side='bottom')
        self.style.configure('Status.TFrame', background=self.colorful_palette['accent'])
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.status_frame, textvariable=self.status_var, 
                  background=self.colorful_palette['accent'], 
                  foreground=self.colorful_palette['button_fg']).pack(side='left', padx=10)
        
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        ttk.Label(self.status_frame, text=f"Device: {device}",
                  background=self.colorful_palette['accent'],
                  foreground=self.colorful_palette['button_fg']).pack(side='right', padx=10)
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def load_image(self):
        """Load and display an image"""
        self.stop_webcam()  # Stop webcam if running
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.img_path = file_path
            self.video_path = None
            self.display_image(file_path)
            self.update_status(f"Loaded: {os.path.basename(file_path)}")
            
            self.brightness_var.set(1.0)
            self.contrast_var.set(1.0)
    
    def load_video(self):
        """Load and process a video file"""
        self.stop_webcam()  # Stop webcam if running
        file_path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        
        if file_path:
            self.video_path = file_path
            self.img_path = None
            self.update_status(f"Loaded video: {os.path.basename(file_path)}")
            threading.Thread(target=self._process_video_thread, daemon=True).start()
    
    def start_webcam(self):
        """Start webcam feed"""
        if self.webcam_active:
            return
        
        self.img_path = None
        self.video_path = None
        try:
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
            self.webcam_active = True
            self.update_status("Webcam started")
            threading.Thread(target=self._process_webcam_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Could not start webcam: {e}")
    
    def stop_webcam(self):
        """Stop webcam feed"""
        if self.webcam_active:
            self.webcam_active = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.update_status("Webcam stopped")
    
    def display_image(self, img_path=None, frame=None):
        """Display image or video frame in canvas"""
        try:
            if img_path:
                image = Image.open(img_path)
            elif frame is not None:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return
            
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                image.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
            else:
                image.thumbnail((800, 600), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(image)
            
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                self.image_canvas.winfo_width()//2, 
                self.image_canvas.winfo_height()//2, 
                anchor='center', 
                image=self.photo
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {e}")
    
    def enhance_image(self, event=None):
        """Apply image enhancements"""
        if not self.img_path or self.video_path or self.webcam_active:
            return
        
        try:
            image = Image.open(self.img_path)
            
            if self.brightness_var.get() != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(self.brightness_var.get())
            
            if self.contrast_var.get() != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(self.contrast_var.get())
            
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                image.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
            else:
                image.thumbnail((800, 600), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(image)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                self.image_canvas.winfo_width()//2, 
                self.image_canvas.winfo_height()//2, 
                anchor='center', 
                image=self.photo
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not enhance image: {e}")
    
    def reset_enhancements(self):
        """Reset image enhancements"""
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        if self.img_path and not self.video_path and not self.webcam_active:
            self.display_image(self.img_path)

    def reset(self):
        self.confidence_var.set(0.4)
        
    
    def detect_disease(self):
        """Detect diseases in the current input"""
        if not (self.img_path or self.video_path or self.webcam_active):
            messagebox.showwarning("Warning", "Please load an image, video, or start webcam first.")
            return
        
        if not self.model:
            messagebox.showerror("Error", "Model not loaded. Please check model path in settings.")
            return
        
        self.update_status("Detecting diseases...")
        
        if self.img_path:
            threading.Thread(target=self._detect_disease_thread, daemon=True).start()
        elif self.video_path:
            threading.Thread(target=self._process_video_thread, daemon=True).start()
        elif self.webcam_active:
            # Webcam detection is handled in _process_webcam_thread
            pass
    
    def _detect_disease_thread(self):
        """Thread function for disease detection on single image"""
        try:
            results = self.model.predict(
                self.img_path, 
                save=True, 
                imgsz=int(self.img_size_var.get()), 
                conf=self.confidence_var.get(),
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            result = results[0]
            
            self.root.after(0, self._clear_results)
            
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                location = f"({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})"
                
                detections.append({
                    'disease': label,
                    'confidence': conf,
                    'location': location,
                    'bbox': (x1, y1, x2, y2)
                })
            
            self.root.after(0, lambda: self._update_detection_results(detections))
            
            if detections:
                save_dir = result.save_dir
                img_file = os.path.join(save_dir, os.path.basename(self.img_path))
                self.root.after(0, lambda: self.display_image(img_file))
            
            detection_record = {
                'timestamp': datetime.now().isoformat(),
                'image_path': self.img_path,
                'detections': detections
            }
            self.detection_history.append(detection_record)
            
            status_msg = f"Detection complete. Found {len(detections)} disease(s)."
            self.root.after(0, lambda: self.update_status(status_msg))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Detection Error", str(e)))
            self.root.after(0, lambda: self.update_status("Detection failed"))
            print(e)
    
    def _process_video_thread(self):
        """Thread function for video processing"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
                
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                results = self.model.predict(
                    frame,
                    imgsz=int(self.img_size_var.get()),
                    conf=self.confidence_var.get(),
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                
                result = results[0]
                detections = []
                
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names[cls_id]
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    location = f"({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})"
                    
                    detections.append({
                        'disease': label,
                        'confidence': conf,
                        'location': location,
                        'bbox': (x1, y1, x2, y2)
                    })
                
                self.root.after(0, self._clear_results)
                self.root.after(0, lambda: self._update_detection_results(detections))
                
                annotated_frame = result.plot()
                self.root.after(0, lambda f=annotated_frame: self.display_image(frame=f))
                
                detection_record = {
                    'timestamp': datetime.now().isoformat(),
                    'image_path': f"video_frame_{datetime.now().timestamp()}",
                    'detections': detections
                }
                self.detection_history.append(detection_record)
                
                self.root.after(0, lambda: self.update_status(f"Processing video frame: {len(self.detection_history)} diseases found"))
                
                cv2.waitKey(33)  # Approximately 30 FPS
                
            cap.release()
            self.root.after(0, lambda: self.update_status("Video processing complete"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Video Processing Error", str(e)))
            self.root.after(0, lambda: self.update_status("Video processing failed"))
    
    def _process_webcam_thread(self):
        """Thread function for webcam processing"""
        try:
            while self.webcam_active and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                results = self.model.predict(
                    frame,
                    imgsz=int(self.img_size_var.get()),
                    conf=self.confidence_var.get(),
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                
                result = results[0]
                detections = []
                
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names[cls_id]
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    location = f"({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})"
                    
                    detections.append({
                        'disease': label,
                        'confidence': conf,
                        'location': location,
                        'bbox': (x1, y1, x2, y2)
                    })
                
                self.root.after(0, self._clear_results)
                self.root.after(0, lambda: self._update_detection_results(detections))
                
                annotated_frame = result.plot()
                self.root.after(0, lambda f=annotated_frame: self.display_image(frame=f))
                
                detection_record = {
                    'timestamp': datetime.now().isoformat(),
                    'image_path': f"webcam_frame_{datetime.now().timestamp()}",
                    'detections': detections
                }
                self.detection_history.append(detection_record)
                
                self.root.after(0, lambda: self.update_status(f"Processing webcam: {len(detections)} diseases found"))
                
                cv2.waitKey(33)  # Approximately 30 FPS
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Webcam Processing Error", str(e)))
            self.root.after(0, lambda: self.update_status("Webcam processing failed"))
    
    def _clear_results(self):
        """Clear detection results"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
    
    def _update_detection_results(self, detections):
        """Update detection results in the treeview"""
        for detection in detections:
            self.results_tree.insert('', 'end', values=(
                detection['disease'],
                f"{detection['confidence']:.3f}",
                detection['location']
            ))
    
    def save_results(self):
        """Save detection results"""
        if not self.detection_history:
            messagebox.showinfo("Info", "No detection results to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.detection_history, f, indent=2)
                self.update_status(f"Results saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save results: {e}")
    
    def select_batch_folder(self):
        """Select folder for batch processing"""
        folder_path = filedialog.askdirectory(title="Select folder containing images")
        
        if folder_path:
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            self.batch_images = []
            
            for file_path in Path(folder_path).rglob('*'):
                if file_path.suffix.lower() in image_extensions:
                    self.batch_images.append(str(file_path))
            
            self.update_status(f"Selected {len(self.batch_images)} images for batch processing")
            
            for item in self.batch_tree.get_children():
                self.batch_tree.delete(item)
    
    def process_batch(self):
        """Process batch of images"""
        if not self.batch_images:
            messagebox.showwarning("Warning", "Please select a folder with images first.")
            return
        
        if not self.model:
            messagebox.showerror("Error", "Model not loaded.")
            return
        
        threading.Thread(target=self._process_batch_thread, daemon=True).start()
    
    def _process_batch_thread(self):
        """Thread function for batch processing"""
        total_images = len(self.batch_images)
        
        for i, img_path in enumerate(self.batch_images):
            try:
                progress = (i / total_images) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda: self.update_status(f"Processing {i+1}/{total_images}"))
                
                results = self.model.predict(
                    img_path, 
                    save=True, 
                    imgsz=int(self.img_size_var.get()), 
                    conf=self.confidence_var.get(),
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                
                result = results[0]
                
                detections = []
                max_conf = 0
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names[cls_id]
                    detections.append(label)
                    max_conf = max(max_conf, conf)
                
                diseases_str = ", ".join(set(detections)) if detections else "None"
                status = "Diseases Found" if detections else "Healthy"
                
                self.root.after(0, lambda p=img_path, s=status, d=diseases_str, c=max_conf: 
                              self.batch_tree.insert('', 'end', values=(
                                  os.path.basename(p), s, d, f"{c:.3f}" if c > 0 else "N/A"
                              )))
                
            except Exception as e:
                self.root.after(0, lambda p=img_path, err=str(e): 
                              self.batch_tree.insert('', 'end', values=(
                                  os.path.basename(p), "Error", err, "N/A"
                              )))
        
        self.root.after(0, lambda: self.progress_var.set(100))
        self.root.after(0, lambda: self.update_status("Batch processing complete"))
    
    def export_batch_report(self):
        """Export batch processing report"""
        if not self.batch_tree.get_children():
            messagebox.showinfo("Info", "No batch results to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerow(['File Name', 'Status', 'Diseases Found', 'Max Confidence'])
                    
                    for item in self.batch_tree.get_children():
                        values = self.batch_tree.item(item, 'values')
                        writer.writerow(values)
                
                self.update_status(f"Batch report exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export report: {e}")
    
    def update_charts(self):
        """Update analysis charts"""
        if not self.detection_history:
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
                ax.text(0.5, 0.5, 'No data available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, color=self.colorful_palette['fg'])
            
            self.canvas.draw()
            return
        
        disease_counts = {}
        confidence_scores = []
        detection_dates = []
        
        for record in self.detection_history:
            date = datetime.fromisoformat(record['timestamp']).date()
            detection_dates.append(date)
            
            for detection in record['detections']:
                disease = detection['disease']
                confidence = detection['confidence']
                
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
                confidence_scores.append(confidence)
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            ax.set_facecolor(self.colorful_palette['select_bg'])
            ax.tick_params(colors=self.colorful_palette['fg'], which='both')
            ax.yaxis.label.set_color(self.colorful_palette['fg'])
            ax.xaxis.label.set_color(self.colorful_palette['fg'])
            ax.title.set_color(self.colorful_palette['fg'])

        if disease_counts:
            diseases = list(disease_counts.keys())
            counts = list(disease_counts.values())
            self.ax1.pie(counts, labels=diseases, autopct='%1.1f%%', textprops={'color': self.colorful_palette['fg']})
            self.ax1.set_title('Disease Distribution')
        
        if confidence_scores:
            self.ax2.hist(confidence_scores, bins=10, alpha=0.7, color=self.colorful_palette['accent'])
            self.ax2.set_xlabel('Confidence Score')
            self.ax2.set_ylabel('Frequency')
            self.ax2.set_title('Confidence Score Distribution')
        
        if detection_dates:
            date_counts = Counter(detection_dates)
            dates = sorted(date_counts.keys())
            counts = [date_counts[date] for date in dates]
            
            self.ax3.plot(dates, counts, marker='o', color=self.colorful_palette['button_bg'])
            self.ax3.set_xlabel('Date')
            self.ax3.set_ylabel('Detections')
            self.ax3.set_title('Detection Timeline')
            self.ax3.tick_params(axis='x', rotation=45)
        
        if disease_counts:
            disease_confidences = {}
            for record in self.detection_history:
                for detection in record['detections']:
                    disease = detection['disease']
                    if disease not in disease_confidences:
                        disease_confidences[disease] = []
                    disease_confidences[disease].append(detection['confidence'])
            
            diseases = list(disease_confidences.keys())
            avg_confidences = [np.mean(disease_confidences[d]) for d in diseases]
            
            self.ax4.bar(diseases, avg_confidences, color=self.colorful_palette['button_bg'])
            self.ax4.set_xlabel('Disease')
            self.ax4.set_ylabel('Average Confidence')
            self.ax4.set_title('Average Confidence by Disease')
            self.ax4.tick_params(axis='x', rotation=45)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_charts(self):
        """Save analysis charts"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.update_status(f"Charts saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save charts: {e}")
    
    def reload_model(self):
        """Reload the YOLO model"""
        model_path = self.model_path_var.get()
        
        try:
            self.model = YOLO(model_path)
            self.update_status(f"Model reloaded from {model_path}")
            messagebox.showinfo("Success", "Model reloaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not reload model: {e}")
            self.model = None
    
    def on_closing(self):
        """Handle application closing"""
        self.stop_webcam()
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    app = AdvancedLeafDiseaseApp(root)
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()