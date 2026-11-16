"""Run Data Designer with a project alias; secrets exist only in process memory."""
from pathlib import Path
import os,sys
R=Path(__file__).resolve().parent
os.environ['DATA_DESIGNER_HOME']=str(R/'designer-config')
os.environ['NVIDIA_API_KEY']=Path.home().joinpath('.config/nvidia/adlr_ngc_api_key').read_text().strip()
os.environ['DATA_DESIGNER_DISABLE_VERSION_CHECK']='1'
os.execv(str(R/'.venv/bin/data-designer'),['data-designer',*sys.argv[1:]])
