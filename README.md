
# mth205 Module Installation and Import Guide

This guide provides instructions on how to install and import the `mth205` module in a Jupyter Notebook. The `mth205` module is designed to support various mathematical operations and is hosted on GitHub. Follow the steps below to install it directly from the repository and use it in your Jupyter Notebook.

## Using mth205 in Jupyter Notebook with Google CoLab

1. **Launch Jupyter Notebook in Google Colab**

2. **Pip install the `mth205` module** at the beginning of your notebook with the following Python code:

```bash
!pip install git+https://github.com/davidaustinm/mth205-W20.git#egg=mth205
```

```python
from mth205 import *
```

Now, you are ready to use the functionalities provided by the `mth205` module in your Jupyter Notebook.

## Prerequisites
- Python 3.6 or higher
- pip (Python package installer)
- Access to a terminal or command prompt
- Jupyter Notebook or JupyterLab installed

If you do not have Jupyter installed, you can install it using pip:

```bash
pip install notebook
```

or if you prefer JupyterLab:

```bash
pip install jupyterlab
```

## Installation

1. **Open your terminal or command prompt**.
   
2. **Navigate to your project directory** where you intend to use the `mth205` module.

3. **Activate your virtual environment** if you are using one. If not, you can skip this step.

4. **Install the `mth205` module** directly from GitHub by running the following command:

```bash
pip install git+https://github.com/jdenhof/mth205-W20.git@pip_installable#egg=mth205
```

This command tells pip to install the `mth205` module from the specified Git repository and branch (`pip_installable`).
The above will need to updated when merged into main repository.

## Using mth205 in Jupyter Notebook locally

After installing the `mth205` module, you can use it in a Jupyter Notebook by following these steps:

1. **Launch Jupyter Notebook** by running `jupyter notebook` or JupyterLab by running `jupyter lab` in your terminal or command prompt.

2. **Create a new notebook** or open an existing one.

3. **Import the `mth205` module** at the beginning of your notebook with the following Python code:

```python
from mth205 import *
```

Now, you are ready to use the functionalities provided by the `mth205` module in your Jupyter Notebook.

## Troubleshooting

If you encounter any issues with installing or importing the `mth205` module, ensure that:

- Your internet connection is stable, as the installation requires downloading the module from GitHub.
- You have the necessary permissions to install Python packages on your system.
- You are using the correct virtual environment where Jupyter and the `mth205` module are installed.

For more information or if you encounter specific issues, please check the GitHub repository's Issues section or contact the repository maintainer.
