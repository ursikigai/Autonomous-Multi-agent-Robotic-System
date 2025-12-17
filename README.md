# **Autonomous Multi-Agent Robotics System with YOLO–SLAM Fusion and Behavior-Based Control**

This project implements an AI-driven multi-agent robotic framework capable of autonomous navigation, dynamic obstacle avoidance, and global mapping. The system integrates perception, SLAM, behavior fusion, local planning, and multi-agent coordination using a centralized controller server.

---

## **1\. Project Structure**

`thesis_multiagent/`  
`├── data/                        # KITTI dataset input (images, poses, lidar)`  
`│   └── kitti/sequences/00/`  
`├── experiments/                 # YOLO outputs, reconstruction, tracking, 3D viewer`  
`├── results/                     # SLAM maps, fusion images, agent plots, videos`  
`│   ├── fusion_tracks.png`  
`│   ├── slam_animation.gif`  
`│   ├── multiagent_plots/`  
`│   ├── plot_min_distances.png`  
`│   ├── risk_histogram.png`  
`│   └── task_density_heatmap.png`  
`├── scripts/`  
`│   └── agent_behavior.py        # Agent logic + behavior fusion`  
`├── server/`  
`│   └── controller.py            # Centralized multi-agent server`  
`├── src/`  
`│   ├── visualization/           # 2D/3D map and trajectory plotting`  
`│   ├── yolo/                    # YOLO detection utilities`  
`│   └── make_slam_mp4.py`  
`└── .venv_thesis311/             # Virtual environment`

---

## **2\. Setting Up the Environment**

### **Step 1: Activate virtual environment**

`cd ~/thesis_multiagent`  
`source .venv_thesis311/bin/activate`

### **Step 2: Install dependencies**

`pip install -r requirements.txt`

---

## **3\. Running the Multi-Agent System**

### **Step 1: Start the central server**

`python -m server.controller`

### **Step 2: Run the agents (open separate terminals)**

`python scripts/agent_behavior.py --agent agent_0 --start_x 0 --start_y 0`  
`python scripts/agent_behavior.py --agent agent_1 --start_x 5 --start_y -2`  
`python scripts/agent_behavior.py --agent agent_2 --start_x -3 --start_y 1.5`

Agents will:  
 • Post their state to the server  
 • Receive goals and obstacle data  
 • Compute behavior mode using the fusion logic  
 • Navigate autonomously

---

## **4\. YOLO Detection on KITTI Dataset**

Run YOLOv8 on KITTI camera frames:

`python src/yolo/run_yolo_kitti.py \`  
    `--source data/kitti/sequences/00/image_0 \`  
    `--out experiments/yolo/kitti_00/annotated`

Outputs include:  
 • Annotated images  
 • Detection CSV file  
 • Tracking visualizations

Annotated frames:

`experiments/yolo/kitti_00/annotated/`

---

## **5\. SLAM Path and Mapping**

Generate SLAM trajectory plot:

`python src/visualization/visualize_slam_path.py \`  
    `--poses data/kitti/sequences/00/poses.txt \`  
    `--out results/slam_topview.png`

Generate 3D interactive SLAM map:

`python src/visualization/visualize_3d_interactive.py`  
`open experiments/yolo/kitti_00/reconstruction/tracking/interactive_3d.html`

---

## **6\. Multi-Agent Behavior Log Visualization**

After running agents, generate plots:

`python src/visualization/plot_agent_logs.py`

Plots saved at:

`results/multiagent_plots/`

Includes:  
 • Agent positions  
 • Behavior modes  
 • Obstacles count  
 • Distance to goal

---

## **7\. Performance Metrics**

Performance visualizations include:  
 • plot\_min\_distances.png  
 • risk\_histogram.png  
 • task\_density\_heatmap.png  
 • rl\_training\_curve.png

Located in:

`results/`

---

## **8\. Fusion Visualization**

Fusion of SLAM trajectory and YOLO detections (selected output):

`results/fusion_tracks.png`

To generate fusion animations or 3D sequences, use:

`python src/make_slam_mp4.py --poses data/kitti/sequences/00/poses.txt --outdir results/`

---

## **9\. Key Features**

• Multi-agent autonomous navigation  
 • Centralized controller with decentralized behavior logic  
 • YOLO-based dynamic obstacle detection  
 • SLAM for global pose estimation  
 • Behavior fusion module for smart decision making  
 • ORCA-inspired local collision avoidance  
 • Real-time visualization tools (2D and 3D)  
 • Scalable to 5–10 agents

---

## **10\. Future Improvements**

• Full 3D LiDAR-based SLAM  
 • Better learned behavior models (PPO / RL-based)  
 • Full decentralized map sharing  
 • Real-world deployment interface  
 • Object-level semantic mapping
