# Environment setup

Use miniconda as reccomended by the developers of mmaction. First, navigate to the mmaction2 folder from the files Anthony uploaded. Move the `requirements_nommaction2.txt` file to this folder. Then use the following commands to make the environment:
1. `conda create --name mmlab2 python=3.9`
2. `conda activate mmlab2`
3. `conda install pip`
4. `pip install -r requirements_gpu.txt`

Pip won't recognize the torch stuff as written in the requirements file, so you have to manually install via:

`pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`

At this point, the actual mmaction package will not be recognized because I removed it from the requirements as the version used by Anthony is no longer supported. Run `pip intall -e .` to install Anthony's modified mmaction2 from the local setup.py file.


### Version fixes
Not sure why requirements.txt does not handle this already, but `numpy.int` has been depreciated, so you will have to replace any `np.int` in the code to `np.int_` (make sure to only replace `np.int`, not `np.int32` or `np.int64`). Also, you have to upgrade mmcv-full via `pip intall --upgrade mmcv-full`. 