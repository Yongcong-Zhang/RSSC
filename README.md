# Rolling Shutter Camera Self-Calibration

Code for **[Rolling Shutter Camera Self-Calibration](https://arxiv.org/pdf/2608.01509)** (ECCV 2026 Oral)

**Contacts:** yongcongzhang0326@gmail.com

![RSL-BA Pipeline](images/pipline.png)

## 📦 Project Page
https://yongcong-zhang.github.io/Rolling-Shutter-Camera-Self-Calibration/

## 📄 Paper
https://arxiv.org/pdf/2608.01509



## Usage

1. **Environment Setup**  
   Clone the repository and install the dependencies listed in `requirements.txt`.

2. **Run Self-Calibration**  
   Open and run the `main.ipynb` Jupyter notebook. It supports all five self-calibration methods proposed in the paper (RSSC-TE, RSSC-CEQ, RSSC-CEH, RSSC-DPQ, RSSC-DPH). You can select and execute the desired method within the notebook.

3. **Using Your Own Data**  
   To calibrate on custom data, organize your images and COLMAP reconstruction outputs following the same directory structure as the provided `datasets/` folder. Then, update the corresponding data paths in `main.ipynb` to point to your dataset.


## Notes

1. The current implementation takes COLMAP reconstruction results as input. For the RSSC-CEQ, RSSC-CEH, RSSC-DPQ, and RSSC-DPH methods, additional feature matching is performed for rectification, which increases computation time. For general use, we recommend using RSSC-TE, which is more efficient.

2. When initializing with COLMAP, please select the **"SIMPLE_PINHOLE"** camera model and enable **"shared for all images"** to improve initialization success rate. Make sure that every image is successfully initialized.


## Citation

If you find this work useful, please cite:

```bibtex
    @article{zhang2026rolling,
      title={Rolling Shutter Camera Self-Calibration},
      author={Zhang, Yongcong and Rabbani, Navid and Liao, Bangyan and Wang, Chengbo and Lao, Yizhen and Bartoli, Adrien},
      journal={arXiv preprint arXiv:2608.01509},
      year={2026}
    }
```
