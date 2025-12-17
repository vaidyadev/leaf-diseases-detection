
# 🌿 Advanced Leaf Disease Detection System (YOLOv8)

## 📌 Project Overview
This project is an **AI-powered leaf disease detection system** built using **YOLOv8** and **Python Tkinter**.  
It detects and classifies plant leaf diseases from **images, videos, batch folders, and live webcam feed** with high accuracy.

The system is designed for **agriculture assistance**, helping farmers, researchers, and students identify crop diseases early.

---

## 🎯 Objectives
- Detect plant leaf diseases accurately using deep learning
- Support real-time detection via webcam and video
- Provide batch processing and result analysis
- Offer an easy-to-use graphical user interface (GUI)

---

## 🧠 Technologies Used
- **Python 3.12**
- **YOLOv8 (Ultralytics)**
- **PyTorch**
- **OpenCV**
- **Tkinter (GUI)**
- **Matplotlib**
- **PIL (Image Processing)**
- **Roboflow (Dataset Management)**

---

## 🌱 Supported Crops & Classes
- **Grapes**
  - Black Measles
  - Black Rot
  - Blight Fungus
  - Healthy
- **Tomato**
- **Rice**
- **Mango**
- **General Plant Model**

*(Auto model switching supported)*

---

## 🚀 Features
✔ Single image disease detection  
✔ Video disease detection  
✔ Live webcam disease detection  
✔ Batch image processing  
✔ Auto crop-specific model switching  
✔ Adjustable confidence threshold  
✔ Image enhancement (brightness & contrast)  
✔ Detection history logging  
✔ Statistical analysis with charts  
✔ Export results (JSON / CSV)  

---

## 🏗️ Project Structure
```
├── models/
│   ├── grapes_best.pt
│   ├── tomato_best.pt
│   ├── rice_best.pt
│   ├── mango_best.pt
│   └── general_best.pt
│
├── diseases.py          # Main application file
├── runs/                # YOLO training & detection outputs
├── README.md            # Project documentation
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/leaf-disease-detection.git
cd leaf-disease-detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
python diseases.py
```

---


# 🌿 How the Leaf Disease Detection Project Works

## 1️⃣ Overall Working Principle

The project is an **AI-based plant leaf disease detection system** that uses a **pre-trained YOLOv8 deep learning model** to identify diseases from plant leaf images, videos, or live webcam feed.

The system follows this flow:

> **Input (Image / Video / Webcam) → YOLOv8 Detection → Result Visualization → Analysis & Export**

---

## 2️⃣ Application Startup

* When the application starts:

  * The **Tkinter GUI** is launched.
  * Multiple **YOLOv8 models** (crop-specific) are loaded from the `models/` folder.
  * A **default model** is selected for initial detection.
  * GPU (CUDA) is automatically used if available.

This ensures the system is **ready for instant detection**.

---

## 3️⃣ Input Handling

The system accepts **four types of input**:

### 🔹 a) Single Image

* User selects a leaf image.
* Image is displayed in the GUI.
* Optional **brightness and contrast adjustments** can be applied before detection.

### 🔹 b) Video File

* A video is loaded and processed **frame-by-frame**.
* Each frame is passed to the detection model.
* Detected diseases are shown live.

### 🔹 c) Live Webcam

* Webcam captures frames in real time.
* Each frame is analyzed instantly.
* Results are displayed continuously.

### 🔹 d) Batch Folder

* User selects a folder containing multiple images.
* All images are processed one by one automatically.
* Summary results are shown in tabular form.

---

## 4️⃣ Disease Detection Process

When the **Detect Disease** button is clicked:

1. The selected input (image/video/webcam frame) is sent to the **YOLOv8 model**.
2. YOLOv8 performs:

   * **Object detection** → finds diseased regions
   * **Classification** → identifies the disease type
3. The model returns:

   * Disease name
   * Confidence score
   * Bounding box coordinates

These results are processed safely using **multi-threading** to keep the UI responsive.

---

## 5️⃣ Auto Model Switching (Smart Feature)

* The system supports **automatic crop-based model switching**.
* If a crop type (e.g., grape or tomato) is detected with high confidence:

  * The application automatically switches to the **best crop-specific model**.
  * Detection is re-run for higher accuracy.

This makes the system **adaptive and intelligent**.

---

## 6️⃣ Result Visualization

After detection:

* Bounding boxes are drawn on the leaf image or video frame.
* Results are shown in a **table**, including:

  * Disease name
  * Confidence score
  * Location of infection
* The annotated image/video frame is displayed in the GUI.

---

## 7️⃣ Detection History & Data Management

* Every detection is stored with:

  * Timestamp
  * Input source
  * Detected diseases and confidence
* History size is automatically limited to avoid memory issues.
* Users can **save results as JSON** for later analysis.

---

## 8️⃣ Batch Processing & Reporting

* Batch images are processed automatically.
* For each image:

  * Disease presence
  * Maximum confidence score
* Results are shown in a table.
* Batch reports can be **exported as CSV files**.

---

## 9️⃣ Analysis & Statistics Module

The system provides an **Analysis tab** that:

* Shows disease distribution
* Displays confidence score trends
* Visualizes detection frequency over time
* Helps users understand disease patterns

Charts can be saved for documentation purposes.

---

## 🔟 User Controls & Settings

Users can:

* Adjust detection confidence threshold
* Select crop-specific models manually
* Enable/disable saving outputs
* Reset parameters anytime

This makes the application **user-friendly and flexible**.

---

## System Shutdown

* Webcam and video streams are safely released.
* Application closes cleanly without resource leaks.

---

##  Final Summary 

> This project works by taking leaf images, videos, or live webcam input and passing them to a YOLOv8-based detection model. The system identifies diseased regions, classifies the disease type, and displays results in real time through an interactive GUI. It supports batch processing, automatic model switching, result analysis, and export features, making it suitable for smart agriculture applications.

---



## 🔮 Future Research Scope

1️⃣ **Mobile Application Development**
- Deploy model using TensorFlow Lite / ONNX
- Android-based disease detection app

2️⃣ **Edge & IoT Integration**
- Deploy on Raspberry Pi / Jetson Nano
- Smart farming systems with cameras

3️⃣ **More Crop & Disease Expansion**
- Add cotton, wheat, maize, potato, etc.
- Support nutrient deficiency detection

4️⃣ **Severity Level Prediction**
- Mild / Moderate / Severe disease grading

5️⃣ **Explainable AI (XAI)**
- Heatmaps (Grad-CAM) to visualize disease regions

6️⃣ **Weather & Soil Data Integration**
- Disease prediction based on environmental factors

7️⃣ **Multilingual Voice Assistance**
- Voice output for farmers in local languages

8️⃣ **Cloud-based Monitoring Dashboard**
- Centralized disease tracking across farms

---

## 👨‍🎓 Academic Use
This project is suitable for:
- Final year engineering projects
- AI / ML / Computer Vision coursework
- Agricultural technology research

---

## 📜 License
This project is developed for **educational and research purposes**.

---

## 🙌 Acknowledgements
- **Ultralytics YOLOv8**
- **Roboflow**
- Open-source Python community
