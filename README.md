## ✅ Features

- Training ML models for predicting traffic conditions
- Read SCATS intersection nodes and draw them on a map
- Load travel-time-based weighted edges from predictions
- Support path finding using various algorithms:
  - A_star, BFS, DFS, GBFS, Custom1, Custom2
- Compute **Top-K shortest paths** with **Yen's algorithm**
- Tkinter-based GUI with interactive node selection and path highlighting
- Dynamic travel time estimation using CNN predictions

## 🛠 How to Train

## 🛠️ How to Train

1. **Process data**  
   python utils/preprocess.py

2. **Train models**  
   python main/model_lstm.py  
   python main/model_gru.py  
   python main/model_cnn.py

3. **Travel time prediction & edge generation**  
   python main/evaluate.py  
   python utils/convert_to_travel_time.py  
   python utils/generate_edges_from_location.py  
   python utils/generate_weighted_edges.py

## 🛠 How to Run

1. Install dependencies (if needed):
   pip install -r requirements.txt

2. Run the GUI interface:
   python gui/main_gui.py
