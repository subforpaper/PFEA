# PFEA Project Guide


## Project Structure

```
PFEA/
├── pfea_framework.py        # Main environment and streamlined pfea framework
├── vlm_planner.py           # VLM-based Chain-of-Objects Reasoning planner
├── vision_utils.py          # Object detection utilities
├── agent_utils.py           # Streamlined agent converter utilities
├── llm_utils.py             # LLM interaction utilities
├── requirements.txt         # Python dependencies
├── ur5e/                    # UR5E robot arm URDF model
│   ├── ur5e.urdf
│   ├── visual/              # Visual meshes (DAE files)
│   └── collision/           # Collision meshes (STL files)
├── robotiq_2f_85/           # Robotiq 2F85 gripper URDF model
│   ├── robotiq_2f_85.urdf
│   └── ...                  # Meshes and textures
└── g_kitting/               # Object templates for manipulation tasks
    ├── object-template.urdf
    └── *.obj                # 3D object models
```


## Installation

1. **Using Conda (Recommended)**:
   ```bash
   # Create conda environment with Python 3.10+
    conda create -n pfea python=3.10 -y
    conda activate pfea
   ```

2. **Install dependencies**:
   ```bash
    pip install -r requirements.txt
   ```

   Main dependencies:
   - `pybullet==3.2.7` - Physics simulation engine
   - `torch==2.4.0` - Deep learning framework
   - `torchvision==0.19.0` - Computer vision utilities
   - `transformers==4.48.1` - HuggingFace transformers
   - `zhipuai==2.1.5` - ZhipuAI API client
   - `opencv-python==4.10.0.84` - Image processing

3. **Configure API Key**:

   Edit `vlm_planner.py` and `llm_utils.py` to set your API key:
   ```python
   API_KEY = "your_api_key_here"
   ```


## Usage

### Quick Start

Run the main simulation:
```bash
python pfea_framework.py
```


## Notes

- **API Key Security**: Do not commit your API key to version control. Consider using environment variables or a separate config file.
- **PyBullet GUI**: The simulation requires a display for GUI mode. For headless servers, use `pybullet.DIRECT` instead of `pybullet.GUI`.
- **Temporary Images**: Camera snapshots are saved as `temp_camera_image_*.png` files during execution.


## Acknowledgments

- [PyBullet](https://pybullet.org/) - Physics engine
- [UR5E Robot Model](https://github.com/ros-industrial/universal_robot) - Robot arm URDF
- [Robotiq 2F85 Gripper](https://github.com/ros-industrial/robotiq) - Gripper URDF
- Open-source VLM and LLM communities
