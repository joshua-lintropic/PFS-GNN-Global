Run the following shell commands [(Adroit)](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch):

```
module load anaconda3/2024.10
conda create --name pfs-gnn python=3.11 -y
conda activate pfs-gnn
pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
pip install torch-geometric
pip install numpy matplotlib tqdm
```
