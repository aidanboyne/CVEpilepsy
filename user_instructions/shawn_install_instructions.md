# Environment setup

Use miniconda as recommended by the developers of mmaction. First, navigate to the mmaction2 folder from the files Anthony uploaded. Move the `requirements_nommaction2.txt` file to this folder. Then use the following commands to make the environment:
1. `conda create --name mmlab2 python=3.9`
2. `conda activate mmlab2`
3. `conda install pip`
4. `pip install -r requirements_nommaction2.txt`

Pip won't recognize the torch stuff as written in the requirements file, so you have to manually install via:

`pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`

At this point, the actual mmaction package will not be recognized because I removed it from the requirements as the version used by Anthony is no longer supported.  
Copy Aidan's copy of mmaction2 as Anthony's folder does not include installation files. Run `pip intall -e .` to install the modified mmaction2 from the local setup.py file.

### Filepaths

Now we have to replace hard-coded filepaths. I'd reccomend using a tool like Notepad++'s find in files or use grep to replace everything quickly. The files that need modification are located in the following paths:
1. `C:\Users\_your.user.name_\mmaction2\i3d_gaussian\i3d_config_XX.py` (where XX is some number)
2. `C:\Users\_your.user.name_\mmaction2\transformer\trans_config_0.py`

In each, replace: `C:/Users/Anth2/Desktop/ML_Seizure_Detection/MMLab` with `C:/Users/_your.user.name_/mmaction2/src`.

### Version fixes
Also, not sure why requirements.txt does not handle this already, but `numpy.int` has been depreciated, so you will have to go through a similar process to replace any `np.int` in the code to `np.int_` (make sure to only replace `np.int`, not `np.int32` or `np.int64`).

### Prepare to upgrade mmcv-full.  You will need CUDA developer toolkit, Visual Studio BuildTools, Visual Studio C++, Visual Studio compiler, Windows10 SDK.
1.  `conda install cudatoolkit-dev=11.6.0 -c conda-forge`
2.  `winget install Microsoft.VisualStudio.2022.BuildTools --force --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK"`
3.  Optional:  `pip install ninja`
Make sure to select proper version of MSVC (highest), SDK (Windows 10 for me), Windows Universal C Runtime.
Now you can compile mmcv-full via `pip install --upgrade mmcv-full`.



### Installation checks
At this point, check if everything is working by running the following: `python install_check.py`. If the environment is correctly installed, you will get something like the result:

```
torch and mmaction are installed successfully
 Results:
[(167, 28.568283), (313, 27.853952), (148, 24.376387), (171, 23.874285), (71, 23.50944)]

```

Additional packages:
`pip install moviepy`
`pip install webcolors`
