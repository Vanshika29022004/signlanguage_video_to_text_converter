# signlanguage_video_to_text_converter

A small project to convert sign language videos to text.

## Python virtual environment

This project uses a local virtual environment named `.venv`.

To create it (already created in this workspace):

 - Windows PowerShell:

```powershell
# create (if not already present)
python -m venv .venv
# activate
. .\.venv\Scripts\Activate.ps1
# deactivate
deactivate
```

If PowerShell execution policy prevents activation, run PowerShell as Administrator and execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or use the built-in `py` launcher:

```powershell
py -3 -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

Once activated, install packages with:

```powershell
pip install -r requirements.txt
```

For questions or issues, open an issue in this repository.
# signlanguage_video_to_text_converter